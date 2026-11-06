#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <ifaddrs.h>
#include <iostream>
#include <limits>
#include <mutex>
#include <net/if.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/socket.h>
#include <sys/types.h>
#include <thread>
#include <utility>
#include <vector>

#include <arpa/inet.h>
#include <unistd.h>

#include "../youtokentome/cpp/third_party/flat_hash_map.h"

namespace {

using Clock = std::chrono::steady_clock;
using vkcom::flat_hash_map;

constexpr size_t kDeltaReserve = 4096;
constexpr size_t kCountWireChunkRecords = 65536;

constexpr char SAMPLE_MAGIC[8] = {'Y', 'T', 'T', 'M', 'I', 'B', 'P', 'E'};
constexpr uint32_t SAMPLE_VERSION = 1;
constexpr char COUNTS_MAGIC[8] = {'Y', 'T', 'T', 'M', 'C', 'N', 'T', '1'};
constexpr char MERGE_MAGIC[8] = {'Y', 'T', 'T', 'M', 'M', 'R', 'G', '1'};

struct Args {
  std::string mode = "single";
  std::string input;
  std::string out_dir;
  int32_t num_merges = 32000;
  int64_t min_frequency = 1000;
  int threads = 8;
  int data_shards = 0;
  int rank = 0;
  int world_size = 1;
  int load_rank = -1;
  int load_world_size = 0;
  int log_every = 100;
  int sync_timeout_sec = 7200;
  uint64_t reserve_pairs = 4000000;
  uint64_t global_reserve_pairs = 0;
  int checkpoint_every = 0;
  std::string resume_merges;
  std::string apply_merges;
  std::string domain_runs;
  std::string encoded_output;
  bool encoded_compact_int = false;
  std::string tcp_rendezvous;
  int tcp_port = 0;
  int32_t apply_skip = 0;
  int32_t token_offset = 0;
  int32_t eos_token_id = 2;
  bool input_shard = false;
  bool sort_positions = true;
  bool use_rle = false;
  bool add_eos = true;
  bool overwrite_output = false;
  bool drop_local_counts_after_init = false;
  bool stream_local_counts = false;
  bool stream_local_deltas = false;
  std::string build_mode = "legacy-noreserve";
};

struct SampleMeta {
  uint64_t nseq = 0;
  uint64_t ntokens = 0;
  int32_t max_token = -1;
};

#ifdef INT_BPE_COMPACT_NODE16
struct __attribute__((packed)) Node {
  uint16_t val = 0;
  int16_t prev = -1;
  int16_t next = -1;
  uint16_t run = 0;
};
static_assert(sizeof(Node) == 8, "compact Node must be exactly 8 bytes");
constexpr uint64_t kNodeValueMax = std::numeric_limits<uint16_t>::max();
constexpr uint64_t kNodeIndexMax = std::numeric_limits<int16_t>::max();
constexpr uint64_t kNodeRunMax = std::numeric_limits<uint16_t>::max();
#else
struct Node {
  int32_t val = 0;
  int32_t prev = -1;
  int32_t next = -1;
  uint32_t run = 0;
};
static_assert(sizeof(Node) == 16, "default Node must stay 16 bytes");
constexpr uint64_t kNodeValueMax = static_cast<uint64_t>(std::numeric_limits<int32_t>::max());
constexpr uint64_t kNodeIndexMax = static_cast<uint64_t>(std::numeric_limits<int32_t>::max());
constexpr uint64_t kNodeRunMax = static_cast<uint64_t>(std::numeric_limits<uint32_t>::max());
#endif

/// 判断链表节点是否仍然代表序列中的有效 token。
inline bool is_alive(const Node &node) {
  return node.run != 0;
}

/// 将节点标记为删除，同时保留原来的数组槽位，避免已有位置索引失效。
inline void mark_dead(Node &node) {
  node.prev = -1;
  node.next = -1;
  node.run = 0;
}

/// 检查紧凑 Node 是否能表示当前样本位置。默认 32 位 Node 下基本不会触发。
void validate_node_index_capacity(uint64_t pos, const std::string &context) {
  if (pos > kNodeIndexMax) {
    throw std::runtime_error(
        "node position exceeds Node index limit in " + context
        + ": pos=" + std::to_string(pos)
        + " max=" + std::to_string(kNodeIndexMax)
        + " node_size_bytes=" + std::to_string(sizeof(Node)));
  }
}

/// 检查紧凑 Node 是否能表示 token id；用于防止 16 位 token 字段静默截断。
void validate_node_value_capacity(int64_t value, const std::string &context) {
  if (value < 0 || static_cast<uint64_t>(value) > kNodeValueMax) {
    throw std::runtime_error(
        "token id exceeds Node value limit in " + context
        + ": value=" + std::to_string(value)
        + " max=" + std::to_string(kNodeValueMax)
        + " node_size_bytes=" + std::to_string(sizeof(Node)));
  }
}

/// 检查紧凑 Node 是否能表示 RLE run 长度。
void validate_node_run_capacity(uint64_t run, const std::string &context) {
  if (run > kNodeRunMax) {
    throw std::runtime_error(
        "run length exceeds Node run limit in " + context
        + ": run=" + std::to_string(run)
        + " max=" + std::to_string(kNodeRunMax)
        + " node_size_bytes=" + std::to_string(sizeof(Node)));
  }
}

#ifdef INT_BPE_COMPACT_POSITION
struct __attribute__((packed)) Position {
  uint32_t seq_id = 0;
  uint16_t pos_id = 0;
};
static_assert(sizeof(Position) == 6, "compact Position must be exactly 6 bytes");
#else
struct Position {
  uint32_t seq_id = 0;
  uint32_t pos_id = 0;
};
static_assert(sizeof(Position) == 8, "default Position must stay 8 bytes");
#endif

/// 构造样本编号；如果样本数超过索引字段容量，直接报错，避免静默截断。
uint32_t make_sequence_id(size_t seq_id) {
  if (seq_id > std::numeric_limits<uint32_t>::max()) {
    throw std::runtime_error(
        "sequence id exceeds uint32_t limit: seq_id=" + std::to_string(seq_id));
  }
  return static_cast<uint32_t>(seq_id);
}

/// 构造 pair 位置索引中的位置记录；字段容量不足时直接报错，避免静默截断。
Position make_position(uint32_t seq_id, size_t pos_id) {
  constexpr uint64_t max_pos = std::numeric_limits<decltype(Position::pos_id)>::max();
  if (static_cast<uint64_t>(pos_id) > max_pos) {
    throw std::runtime_error(
        "position id exceeds position index limit: seq_id=" + std::to_string(seq_id)
        + " pos_id=" + std::to_string(pos_id)
        + " max_pos_id=" + std::to_string(max_pos)
        + " position_size_bytes=" + std::to_string(sizeof(Position)));
  }
  return {seq_id, static_cast<decltype(Position::pos_id)>(pos_id)};
}

/// 检查样本长度是否能被当前编译出的 Position 结构表示。
///
/// 默认版 Position 使用 32 位位置字段；compact 版使用 16 位位置字段来节省
/// pair 位置索引内存。如果单条序列超过字段上限，这里会直接报错，防止把
/// 位置截断后继续训练出错误 codebook。
void validate_sequence_length(uint32_t len, uint64_t seq_index, const std::string &path) {
  constexpr uint64_t max_pos = std::numeric_limits<decltype(Position::pos_id)>::max();
  if (len > 0 && static_cast<uint64_t>(len - 1) > max_pos) {
    throw std::runtime_error(
        "sequence length exceeds position index limit in " + path
        + ": seq_index=" + std::to_string(seq_index)
        + " len=" + std::to_string(len)
        + " max_position_id=" + std::to_string(max_pos)
        + " position_size_bytes=" + std::to_string(sizeof(Position)));
  }
  if (len > 0) {
    validate_node_index_capacity(
        static_cast<uint64_t>(len - 1),
        path + " seq_index=" + std::to_string(seq_index));
  }
}

/// 检查训练期间可能产生的最大 token id 是否能被当前 Node 表示。
void validate_training_token_capacity(int32_t max_token, int32_t num_merges, const std::string &path) {
  const int64_t max_created_token = static_cast<int64_t>(max_token) + static_cast<int64_t>(num_merges);
  validate_node_value_capacity(max_token, path + " input max_token");
  validate_node_value_capacity(max_created_token, path + " max token after requested merges");
}

struct Candidate {
  uint64_t key = 0;
  int64_t count = 0;
};

struct CountWire {
  uint64_t key = 0;
  int64_t value = 0;
};
static_assert(sizeof(CountWire) == 16, "CountWire must stay wire-compatible");

struct MergeRecord {
  int32_t step = 0;
  int32_t left = 0;
  int32_t right = 0;
  int32_t new_id = 0;
  int64_t freq = 0;
  uint64_t real_merges = 0;
  uint64_t positions_seen = 0;
};

struct StepProfile {
  int32_t step = 0;
  int32_t left = 0;
  int32_t right = 0;
  int32_t new_id = 0;
  int64_t freq = 0;
  uint64_t real_merges = 0;
  uint64_t positions_seen = 0;
  uint64_t shard_delta_unique_sum = 0;
  uint64_t local_delta_unique = 0;
  uint64_t global_unique = 0;
  double select_sec = 0.0;
  double merge_sync_sec = 0.0;
  double apply_wall_sec = 0.0;
  double worker_apply_sum_sec = 0.0;
  double worker_apply_max_sec = 0.0;
  double sort_sum_sec = 0.0;
  double local_delta_build_sec = 0.0;
  double delta_send_sec = 0.0;
  double delta_local_apply_sec = 0.0;
  double delta_recv_apply_sec = 0.0;
  double delta_reduce_sec = 0.0;
  double step_wall_sec = 0.0;
};

struct StepStats {
  int32_t step = 0;
  int32_t left = 0;
  int32_t right = 0;
  int32_t new_id = 0;
  int64_t freq = 0;
  uint64_t real_merges = 0;
  uint64_t positions_seen = 0;
  uint64_t changed_unique_pairs = 0;
  uint64_t tokens_after = 0;
  double apply_sec = 0.0;
  double sort_sec = 0.0;
};

struct Summary {
  std::string mode;
  int rank = 0;
  int world_size = 1;
  int threads = 1;
  int data_shards = 1;
  bool input_shard = false;
  int load_rank = 0;
  int load_world_size = 1;
  int32_t num_merges_requested = 0;
  int32_t num_merges_performed = 0;
  int64_t min_frequency = 0;
  bool use_rle = false;
  bool sort_positions = false;
  std::string build_mode;
  bool pq_min_frequency_filter = true;
  uint64_t count_wire_chunk_records = kCountWireChunkRecords;
  uint64_t node_size_bytes = sizeof(Node);
  uint64_t position_size_bytes = sizeof(Position);
  uint64_t reserve_pairs = 0;
  uint64_t global_reserve_pairs = 0;
  int checkpoint_every = 0;
  std::string resume_merges;
  int32_t resume_steps = 0;
  bool drop_local_counts_after_init = false;
  bool stream_local_counts = false;
  bool stream_local_deltas = false;
  uint64_t input_sequences = 0;
  uint64_t input_tokens = 0;
  uint64_t final_tokens = 0;
  uint64_t encoded_records = 0;
  uint64_t encoded_tokens = 0;
  int32_t encoded_min_token = std::numeric_limits<int32_t>::max();
  int32_t encoded_max_token = -1;
  int32_t token_offset = 0;
  int32_t eos_token_id = 2;
  bool add_eos = false;
  bool encoded_compact_int = false;
  std::string encoded_output;
  uint64_t initial_unique_pairs = 0;
  uint64_t final_unique_pairs = 0;
  uint64_t apply_positions_seen = 0;
  uint64_t apply_real_merges = 0;
  double connect_sec = 0.0;
  double load_sec = 0.0;
  double split_sec = 0.0;
  double build_sec = 0.0;
  double build_worker_sum_sec = 0.0;
  double build_worker_max_sec = 0.0;
  double build_node_sum_sec = 0.0;
  double build_node_max_sec = 0.0;
  double build_index_sum_sec = 0.0;
  double build_index_max_sec = 0.0;
  double local_initial_sec = 0.0;
  double initial_reduce_sec = 0.0;
  double barrier_sec = 0.0;
  double select_sec = 0.0;
  double apply_sec = 0.0;
  double sort_sec = 0.0;
  double delta_reduce_sec = 0.0;
  double sync_wait_sec = 0.0;
  double write_sec = 0.0;
  double train_sec = 0.0;
  double total_sec = 0.0;
  uint64_t network_bytes_sent = 0;
  uint64_t network_bytes_recv = 0;
  std::vector<MergeRecord> merges;
  std::vector<StepProfile> profiles;
};

struct DomainStat {
  std::string name;
  uint64_t records = 0;
  uint64_t input_tokens = 0;
  uint64_t output_tokens = 0;
  uint64_t max_input_len = 0;
  uint64_t max_output_len = 0;
};

std::vector<DomainStat> read_domain_runs(const std::string &path);

/// 将一对 int32 token 打包成稳定的 64 位 key，供 hash map 和优先队列使用。
uint64_t pack_pair(int32_t left, int32_t right) {
  return (static_cast<uint64_t>(static_cast<uint32_t>(left)) << 32u)
      | static_cast<uint32_t>(right);
}

/// 从打包后的 pair key 中取出左侧 token id。
int32_t unpack_left(uint64_t key) {
  return static_cast<int32_t>(key >> 32u);
}

/// 从打包后的 pair key 中取出右侧 token id。
int32_t unpack_right(uint64_t key) {
  return static_cast<int32_t>(key & 0xffffffffu);
}

/// 计算从给定时间点到当前经过的秒数，用于日志和 profile。
double seconds_since(const Clock::time_point &start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}

/// 向日志流写入进度百分比、每秒 step 数和 ETA，方便观察长任务进度。
void write_progress_fields(std::ostream &out, int32_t completed_steps, int32_t total_steps, double elapsed_sec) {
  const double progress = total_steps > 0 ? static_cast<double>(completed_steps) / static_cast<double>(total_steps) : 0.0;
  const double step_per_sec = elapsed_sec > 0.0 ? static_cast<double>(completed_steps) / elapsed_sec : 0.0;
  const double eta_sec = step_per_sec > 0.0 && total_steps > completed_steps
      ? static_cast<double>(total_steps - completed_steps) / step_per_sec
      : 0.0;
  out << " progress_pct=" << std::fixed << std::setprecision(2) << (progress * 100.0)
      << " step_per_sec=" << std::setprecision(4) << step_per_sec
      << " eta_sec=" << std::setprecision(1) << eta_sec
      << std::defaultfloat;
}

/// 比较两个 BPE 候选 pair：先看频率，频率相同时选更小的 pair key。
bool better_pair(int64_t freq, uint64_t key, int64_t best_freq, uint64_t best_key) {
  if (freq != best_freq) return freq > best_freq;
  return key < best_key;
}

struct CandidateLess {
  /// 定义候选优先队列顺序，使最高频、同频 key 最小的 pair 先弹出。
  bool operator()(const Candidate &a, const Candidate &b) const {
    if (a.count != b.count) return a.count < b.count;
    return a.key > b.key;
  }
};

/// 计算一个 RLE run 内部包含多少个相邻同 token pair。
uint64_t pairs_in_run(uint32_t run) {
  if (run < 2) return 0;
  return static_cast<uint64_t>(run - 1);
}

/// 计算近似 vector 自动扩容后的容量，用于模拟历史构建模式的内存行为。
size_t vector_growth_capacity(size_t size) {
  if (size <= 1) return size;
  size_t cap = 1;
  while (cap < size && cap <= std::numeric_limits<size_t>::max() / 2) {
    cap *= 2;
  }
  return cap < size ? size : cap;
}

/// 根据构建模式决定每条序列的 Node 数组预留容量。
size_t node_reserve_capacity(size_t size, const std::string &build_mode) {
  if (build_mode == "legacy-growth-reserve" || build_mode == "onepass-growth-reserve") {
    return vector_growth_capacity(size);
  }
  if (build_mode == "legacy-reserve" || build_mode == "onepass") {
    return size;
  }
  return 0;
}

/// 释放已经完成构建的原始 int 序列，降低构建阶段的峰值内存。
///
/// BPE 训练真正使用的是 seqs 里的链表节点和 pair 索引；一条样本完成
/// Node 构建和初始 pair 统计之后，原始 int token 数组就不会再被访问。
/// 这里用 swap 确保 vector capacity 也归还给 allocator，而不是只把 size 清零。
void release_source_sequence(std::vector<int32_t> &seq) {
  std::vector<int32_t>().swap(seq);
}

/// 创建输出目录；目录已存在时保持不变。
void ensure_dir(const std::string &path) {
  std::string cmd = "mkdir -p '" + path + "'";
  if (std::system(cmd.c_str()) != 0) {
    throw std::runtime_error("failed to create directory: " + path);
  }
}

/// 尽力删除临时文件，主要用于清理分布式同步文件。
void remove_file(const std::string &path) {
  std::remove(path.c_str());
}

/// 原子替换文件：先写临时文件，再 rename 到目标路径，避免 checkpoint 半写入。
void atomic_rename(const std::string &tmp, const std::string &path) {
  if (std::rename(tmp.c_str(), path.c_str()) != 0) {
    throw std::runtime_error("failed to rename " + tmp + " to " + path + ": " + std::strerror(errno));
  }
}

/// 检查路径是否存在且能以二进制读方式打开。
bool file_exists(const std::string &path) {
  std::ifstream in(path, std::ios::binary);
  return in.good();
}

/// 轮询等待某个文件出现，并累计等待时间。
void wait_for_file(const std::string &path, int timeout_sec, double &wait_accum) {
  auto start = Clock::now();
  while (!file_exists(path)) {
    if (seconds_since(start) > timeout_sec) {
      throw std::runtime_error("timed out waiting for " + path);
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  wait_accum += seconds_since(start);
}

/// 拼接两个路径片段，避免重复插入斜杠。
std::string join_path(const std::string &a, const std::string &b) {
  if (a.empty()) return b;
  if (a.back() == '/') return a + b;
  return a + "/" + b;
}

template <typename T>
/// 将一个普通二进制值直接写入文件流。
void write_raw(std::ofstream &out, const T &value) {
  out.write(reinterpret_cast<const char *>(&value), sizeof(T));
}

template <typename T>
/// 从文件流读取一个普通二进制值，读不完整时直接报错。
void read_raw(std::ifstream &in, T &value) {
  in.read(reinterpret_cast<char *>(&value), sizeof(T));
  if (!in) throw std::runtime_error("unexpected EOF");
}

/// 写 compact-int 样本文件头；该格式可直接作为后续 int BPE 输入。
void write_sample_header(std::ofstream &out, uint64_t nseq, uint64_t ntokens, int32_t max_token) {
  out.write(SAMPLE_MAGIC, 8);
  const uint32_t version = SAMPLE_VERSION;
  write_raw(out, version);
  write_raw(out, nseq);
  write_raw(out, ntokens);
  write_raw(out, max_token);
}

/// 读取并校验 compact int FEuler 样本文件头。
SampleMeta read_sample_meta(std::ifstream &in, const std::string &path) {
  char magic[8];
  in.read(magic, 8);
  if (!in || !std::equal(magic, magic + 8, SAMPLE_MAGIC)) {
    throw std::runtime_error("invalid int sample magic: " + path);
  }
  uint32_t version = 0;
  read_raw(in, version);
  if (version != SAMPLE_VERSION) {
    throw std::runtime_error(
        "unsupported int sample version in " + path + ": version=" + std::to_string(version));
  }
  SampleMeta meta;
  read_raw(in, meta.nseq);
  read_raw(in, meta.ntokens);
  read_raw(in, meta.max_token);
  if (meta.nseq > std::numeric_limits<uint32_t>::max()) {
    throw std::runtime_error(
        "sample sequence count exceeds uint32_t position index limit in " + path
        + ": nseq=" + std::to_string(meta.nseq));
  }
  if (meta.max_token < 0) {
    throw std::runtime_error("sample max_token must be non-negative: " + path);
  }
  validate_node_value_capacity(meta.max_token, path + " input max_token");
  return meta;
}

/// 读取 compact int 序列；分布式模式下只加载当前 rank 负责的样本。
std::vector<std::vector<int32_t>> load_sample(const std::string &path, int rank, int world_size, SampleMeta &meta) {
  auto start = Clock::now();
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("failed to open sample: " + path);
  if (rank < 0 || world_size <= 0 || rank >= world_size) {
    throw std::runtime_error(
        "invalid load partition for " + path
        + ": rank=" + std::to_string(rank)
        + " world_size=" + std::to_string(world_size));
  }
  meta = read_sample_meta(in, path);
  std::vector<std::vector<int32_t>> seqs;
  seqs.reserve(static_cast<size_t>((meta.nseq + world_size - 1) / world_size));
  uint64_t observed_tokens = 0;
  for (uint64_t i = 0; i < meta.nseq; ++i) {
    uint32_t len = 0;
    read_raw(in, len);
    validate_sequence_length(len, i, path);
    observed_tokens += len;
    if (world_size <= 1 || static_cast<int>(i % static_cast<uint64_t>(world_size)) == rank) {
      std::vector<int32_t> seq(len);
      if (len > 0) {
        in.read(reinterpret_cast<char *>(seq.data()), static_cast<std::streamsize>(len * sizeof(int32_t)));
        if (!in) {
          throw std::runtime_error(
              "unexpected EOF in sequence tokens: " + path + " seq_index=" + std::to_string(i));
        }
      }
      seqs.emplace_back(std::move(seq));
    } else {
      in.seekg(static_cast<std::streamoff>(len) * static_cast<std::streamoff>(sizeof(int32_t)), std::ios::cur);
      if (!in) {
        throw std::runtime_error(
            "failed to seek over sequence: " + path + " seq_index=" + std::to_string(i));
      }
    }
  }
  if (observed_tokens != meta.ntokens) {
    throw std::runtime_error(
        "sample token count mismatch in " + path
        + ": header_ntokens=" + std::to_string(meta.ntokens)
        + " observed_ntokens=" + std::to_string(observed_tokens));
  }
  std::cerr << "loaded sample rank=" << rank << "/" << world_size
            << " seqs=" << seqs.size() << " elapsed_sec=" << seconds_since(start) << std::endl;
  return seqs;
}

/// 计算当前进程实际从输入文件中读取哪一份样本。
///
/// 常规分布式输入有两种：
/// 1. 单个输入文件被所有 rank 按样本编号取模切分；
/// 2. 每个 rank 已经拿到独立 shard，此时默认完整读取自己的 shard。
///
/// 当一个已有 shard 仍然太大时，可以用 --load-rank/--load-world-size
/// 在这个 shard 内再按样本编号取模切分。这样不会切开单条样本，也不会改变
/// 全局 BPE 统计语义，只是把同一个 shard 的样本分给更多 worker。
std::pair<int, int> effective_load_partition(const Args &args) {
  if (args.load_rank >= 0 || args.load_world_size > 0) {
    if (args.load_rank < 0 || args.load_world_size <= 0) {
      throw std::runtime_error("--load-rank and --load-world-size must be set together");
    }
    if (args.load_rank >= args.load_world_size) {
      throw std::runtime_error("invalid --load-rank/--load-world-size");
    }
    return {args.load_rank, args.load_world_size};
  }
  if (args.input_shard) {
    return {0, 1};
  }
  return {args.rank, args.world_size};
}

struct Shard {
  std::vector<std::vector<Node>> seqs;
  flat_hash_map<uint64_t, std::vector<Position>> pair2pos;
  flat_hash_map<uint64_t, int64_t> pair2cnt;
  flat_hash_map<uint64_t, int64_t> delta;
  bool pair_counts_dropped = false;
  uint64_t token_count = 0;
  uint64_t position_offset = 0;
  double build_node_sec = 0.0;
  double build_index_sec = 0.0;

  /// 预留 shard 内 pair 位置表、计数表和 delta 表的容量。
  void reserve(uint64_t reserve_pairs) {
    pair2pos.reserve(reserve_pairs);
    pair2cnt.reserve(reserve_pairs);
    delta.reserve(kDeltaReserve);
  }

  /// 释放初始 pair 计数表，仅保留 pair 位置索引。
  ///
  /// 训练开始后，每轮更新全局频率只依赖本轮 delta；本地 apply 查的是
  /// pair2pos，不再读取 pair2cnt。因此初始计数完成汇总后可以释放它，
  /// 用于降低每个 rank 的常驻内存。
  void drop_pair_counts() {
    flat_hash_map<uint64_t, int64_t>().swap(pair2cnt);
    pair_counts_dropped = true;
  }

  /// 对当前 shard 的非零初始 pair 计数逐条回调，避免额外构造大聚合表。
  template <typename Fn>
  void for_each_pair_count(Fn &&fn) const {
    for (const auto &item : pair2cnt) {
      if (item.second != 0) fn(item.first, item.second);
    }
  }

  /// 对本轮非零 delta 逐条回调，供网络发送或全局 reduce 使用。
  template <typename Fn>
  void for_each_delta(Fn &&fn) const {
    for (const auto &item : delta) {
      if (item.second != 0) fn(item.first, item.second);
    }
  }

  /// 读取某个有效节点及其后继节点组成的当前 pair key。
  uint64_t current_pair_key(uint32_t seq_id, int32_t pos_id) const {
    const auto &nodes = seqs[seq_id];
    const Node &a = nodes[pos_id];
    const Node &b = nodes[a.next];
    return pack_pair(a.val, b.val);
  }

  /// 对某个 pair 增加一个有符号计数变化，并按需记录到本轮 delta。
  void add_count(uint64_t key, int64_t amount, bool record_delta) {
    // amount==0 是正常的无操作路径，不属于错误输入。
    if (amount == 0) return;
    if (!pair_counts_dropped) {
      auto inserted = pair2cnt.emplace(key, amount);
      if (!inserted.second) {
        inserted.first->second += amount;
      }
    }
    if (record_delta) {
      delta[key] += amount;
    }
  }

  /// 将某个节点开头的相邻 pair 加入位置索引，并增加对应计数。
  void add_pair(uint32_t seq_id, int32_t pos_id, bool record_delta) {
    Node &a = seqs[seq_id][pos_id];
    // pair 索引是增量维护的；邻域更新时可能再次访问已经被 merge 消掉的节点。
    // 这些 stale 位置必须跳过，否则会破坏 BPE 的非重叠 merge 语义。
    if (!is_alive(a) || a.next < 0) return;
    Node &b = seqs[seq_id][a.next];
    if (!is_alive(b)) return;
    uint64_t key = pack_pair(a.val, b.val);
    pair2pos[key].push_back(make_position(seq_id, static_cast<size_t>(pos_id)));
    add_count(key, 1, record_delta);
  }

  /// 将一个 RLE 节点内部的重复同 token pair 加入索引和计数。
  void add_self_pair(uint32_t seq_id, int32_t pos_id, bool record_delta) {
    Node &a = seqs[seq_id][pos_id];
    // run 长度不足 2 时没有 self-pair，这是正常无操作路径。
    if (!is_alive(a) || a.run < 2) return;
    uint64_t key = pack_pair(a.val, a.val);
    pair2pos[key].push_back(make_position(seq_id, static_cast<size_t>(pos_id)));
    add_count(key, static_cast<int64_t>(pairs_in_run(a.run)), record_delta);
  }

  /// 减少某个节点开头的当前相邻 pair 计数。
  void remove_pair(uint32_t seq_id, int32_t pos_id) {
    Node &a = seqs[seq_id][pos_id];
    // 删除邻域 pair 时可能遇到已经被本轮 merge 消掉的节点；这类 stale 位置跳过。
    if (!is_alive(a) || a.next < 0) return;
    Node &b = seqs[seq_id][a.next];
    if (!is_alive(b)) return;
    add_count(pack_pair(a.val, b.val), -1, true);
  }

  /// 减少一个 RLE 节点内部同 token pair 的计数。
  void remove_self_pair(uint32_t seq_id, int32_t pos_id) {
    Node &a = seqs[seq_id][pos_id];
    // run 长度不足 2 时没有 self-pair，这是正常无操作路径。
    if (!is_alive(a) || a.run < 2) return;
    add_count(pack_pair(a.val, a.val), -static_cast<int64_t>(pairs_in_run(a.run)), true);
  }

  /// 当跨 run 边界的 merge 消耗一个 token 时，缩短对应 RLE run。
  void decrement_run(uint32_t seq_id, int32_t pos_id) {
    Node &a = seqs[seq_id][pos_id];
    if (!is_alive(a) || a.run < 2) throw std::runtime_error("invalid run decrement");
    a.run -= 1;
    token_count -= 1;
    add_count(pack_pair(a.val, a.val), -1, true);
  }

  /// BPE 更新后，如果相邻节点 token 变相同，则把它们重新合并为一个 RLE run。
  void maybe_coalesce_adjacent(uint32_t seq_id, int32_t left_pos) {
    auto &nodes = seqs[seq_id];
    while (left_pos >= 0 && static_cast<size_t>(left_pos) < nodes.size()) {
      Node &left = nodes[left_pos];
      // 只有相邻 token 相同时才需要合并 run；其他情况是正常退出条件。
      if (!is_alive(left) || left.next < 0) return;
      const int32_t right_pos = left.next;
      Node &right = nodes[right_pos];
      if (!is_alive(right) || left.val != right.val) return;

      remove_pair(seq_id, left_pos);
      remove_self_pair(seq_id, left_pos);
      remove_self_pair(seq_id, right_pos);
      if (right.next >= 0) remove_pair(seq_id, right_pos);

      const uint64_t merged_run = static_cast<uint64_t>(left.run) + static_cast<uint64_t>(right.run);
      validate_node_run_capacity(merged_run, "maybe_coalesce_adjacent");
      left.run = static_cast<decltype(left.run)>(merged_run);
      left.next = right.next;
      mark_dead(right);
      if (left.next >= 0) {
        nodes[left.next].prev = left_pos;
        add_pair(seq_id, left_pos, true);
      }
      add_self_pair(seq_id, left_pos, true);
    }
  }

  /// 从整数序列构建当前 shard 的链表表示和初始 pair 索引。
  void build_from_sequences(std::vector<std::vector<int32_t>> input,
                            uint64_t reserve_pairs,
                            bool use_rle,
                            const std::string &build_mode,
                            int32_t base_vocab_size) {
    reserve(reserve_pairs);
    seqs.reserve(input.size());
    if (!use_rle && build_mode == "base-bucket") {
      if (base_vocab_size <= 0) {
        throw std::runtime_error("base-bucket build mode requires a positive base vocab size");
      }
      const uint64_t slot_count64 = static_cast<uint64_t>(base_vocab_size) * static_cast<uint64_t>(base_vocab_size);
      if (slot_count64 > 1000000ULL) {
        throw std::runtime_error("base-bucket build mode requires a compact base vocab");
      }
      const size_t slot_count = static_cast<size_t>(slot_count64);
      std::vector<std::vector<Position>> base_positions(slot_count);
      std::vector<int64_t> base_counts(slot_count, 0);
      flat_hash_map<uint64_t, std::vector<Position>> overflow_positions;
      flat_hash_map<uint64_t, int64_t> overflow_counts;
      overflow_positions.reserve(1024);
      overflow_counts.reserve(1024);

      for (auto &seq : input) {
        const uint32_t sid = make_sequence_id(seqs.size());
        std::vector<Node> nodes;
        auto node_start = Clock::now();
        for (int32_t token : seq) {
          const int32_t id = static_cast<int32_t>(nodes.size());
          validate_node_index_capacity(static_cast<uint64_t>(id), "base-bucket build");
          Node node;
          node.val = token;
          node.prev = id == 0 ? -1 : id - 1;
          node.next = -1;
          node.run = 1;
          if (!nodes.empty()) nodes.back().next = id;
          nodes.push_back(node);
        }
        build_node_sec += seconds_since(node_start);
        auto index_start = Clock::now();
        for (size_t i = 1; i < seq.size(); ++i) {
          const int32_t left = seq[i - 1];
          const int32_t right = seq[i];
          if (left >= 0 && right >= 0 && left < base_vocab_size && right < base_vocab_size) {
            const size_t bucket = static_cast<size_t>(left) * static_cast<size_t>(base_vocab_size)
                + static_cast<size_t>(right);
            base_positions[bucket].push_back(make_position(sid, i - 1));
            base_counts[bucket] += 1;
          } else {
            const uint64_t key = pack_pair(left, right);
            overflow_positions[key].push_back(make_position(sid, i - 1));
            overflow_counts[key] += 1;
          }
        }
        build_index_sec += seconds_since(index_start);
        token_count += seq.size();
        seqs.emplace_back(std::move(nodes));
        release_source_sequence(seq);
      }

      auto map_start = Clock::now();
      for (int32_t left = 0; left < base_vocab_size; ++left) {
        for (int32_t right = 0; right < base_vocab_size; ++right) {
          const size_t bucket = static_cast<size_t>(left) * static_cast<size_t>(base_vocab_size)
              + static_cast<size_t>(right);
          if (base_counts[bucket] == 0) continue;
          const uint64_t key = pack_pair(left, right);
          pair2cnt.emplace(key, base_counts[bucket]);
          pair2pos.emplace(key, std::move(base_positions[bucket]));
        }
      }
      for (auto &item : overflow_counts) {
        pair2cnt.emplace(item.first, item.second);
      }
      for (auto &item : overflow_positions) {
        pair2pos.emplace(item.first, std::move(item.second));
      }
      build_index_sec += seconds_since(map_start);
      return;
    }
    if (!use_rle && (build_mode == "onepass" ||
                     build_mode == "onepass-noreserve" ||
                     build_mode == "onepass-growth-reserve")) {
      for (auto &seq : input) {
        const uint32_t sid = make_sequence_id(seqs.size());
        std::vector<Node> nodes;
        const size_t reserve_cap = node_reserve_capacity(seq.size(), build_mode);
        if (reserve_cap > 0) nodes.reserve(reserve_cap);
        auto node_start = Clock::now();
        for (size_t i = 0; i < seq.size(); ++i) {
          const int32_t id = static_cast<int32_t>(i);
          validate_node_index_capacity(static_cast<uint64_t>(id), "onepass build");
          Node node;
          node.val = seq[i];
          node.prev = id == 0 ? -1 : id - 1;
          node.next = i + 1 < seq.size() ? id + 1 : -1;
          node.run = 1;
          nodes.push_back(node);
        }
        build_node_sec += seconds_since(node_start);
        auto index_start = Clock::now();
        for (size_t i = 1; i < seq.size(); ++i) {
          if (i > 0) {
            const uint64_t key = pack_pair(seq[i - 1], seq[i]);
            pair2pos[key].push_back(make_position(sid, i - 1));
            pair2cnt[key] += 1;
          }
        }
        build_index_sec += seconds_since(index_start);
        token_count += seq.size();
        seqs.emplace_back(std::move(nodes));
        release_source_sequence(seq);
      }
      return;
    }
    for (auto &seq : input) {
      const uint32_t sid = make_sequence_id(seqs.size());
      std::vector<Node> nodes;
      const size_t reserve_cap = node_reserve_capacity(seq.size(), build_mode);
      if (reserve_cap > 0) nodes.reserve(reserve_cap);
      auto node_start = Clock::now();
      for (int32_t token : seq) {
        if (use_rle && !nodes.empty() && nodes.back().val == token) {
          const uint64_t next_run = static_cast<uint64_t>(nodes.back().run) + 1;
          validate_node_run_capacity(next_run, "rle build");
          nodes.back().run = static_cast<decltype(nodes.back().run)>(next_run);
        } else {
          const int32_t id = static_cast<int32_t>(nodes.size());
          validate_node_index_capacity(static_cast<uint64_t>(id), "legacy build");
          Node node;
          node.val = token;
          node.prev = id == 0 ? -1 : id - 1;
          node.next = -1;
          node.run = 1;
          if (!nodes.empty()) nodes.back().next = id;
          nodes.push_back(node);
        }
      }
      build_node_sec += seconds_since(node_start);
      token_count += seq.size();
      seqs.emplace_back(std::move(nodes));
      auto &built = seqs.back();
      auto index_start = Clock::now();
      for (size_t i = 0; i < built.size(); ++i) {
        if (built[i].run >= 2) add_self_pair(sid, static_cast<int32_t>(i), false);
        if (built[i].next >= 0) add_pair(sid, static_cast<int32_t>(i), false);
      }
      build_index_sec += seconds_since(index_start);
      release_source_sequence(seq);
    }
  }

  /// 在一个 RLE run 内按从左到右、非重叠语义应用 (token, token) merge。
  void apply_self_merge(uint32_t seq_id, int32_t pos_id, int32_t new_id, StepStats &stats) {
    auto &nodes = seqs[seq_id];
    Node &a = nodes[pos_id];
    // pair2pos 里允许保留旧位置；真正 apply 时再次确认当前位置仍匹配。
    if (!is_alive(a) || a.run < 2) return;

    const int32_t p0 = a.prev;
    const int32_t p3 = a.next;
    remove_self_pair(seq_id, pos_id);
    if (p0 >= 0) remove_pair(seq_id, p0);
    if (p3 >= 0) remove_pair(seq_id, pos_id);

    const uint32_t old_run = a.run;
    const uint32_t merges = old_run / 2;
    if (old_run % 2 == 0) {
      a.val = new_id;
      validate_node_run_capacity(merges, "apply_self_merge even");
      a.run = merges;
      a.next = p3;
      if (p3 >= 0) nodes[p3].prev = pos_id;
      if (p0 >= 0) add_pair(seq_id, p0, true);
      if (p3 >= 0) add_pair(seq_id, pos_id, true);
      if (a.run >= 2) add_self_pair(seq_id, pos_id, true);
    } else {
      Node tail;
      tail.val = a.val;
      tail.prev = pos_id;
      tail.next = p3;
      tail.run = 1;
      const int32_t tail_pos = static_cast<int32_t>(nodes.size());
      validate_node_index_capacity(static_cast<uint64_t>(tail_pos), "apply_self_merge tail");
      nodes.push_back(tail);
      Node &cur = nodes[pos_id];
      cur.val = new_id;
      validate_node_run_capacity(merges, "apply_self_merge odd");
      cur.run = merges;
      cur.next = tail_pos;
      if (p3 >= 0) nodes[p3].prev = tail_pos;
      if (p0 >= 0) add_pair(seq_id, p0, true);
      add_pair(seq_id, pos_id, true);
      if (p3 >= 0) add_pair(seq_id, tail_pos, true);
      if (cur.run >= 2) add_self_pair(seq_id, pos_id, true);
    }
    token_count -= merges;
    stats.real_merges += merges;
  }

  /// 对一个有效相邻 pair 应用 merge，并同步更新左右邻域的 pair 计数。
  void apply_pair_merge(uint32_t seq_id, int32_t p1, int32_t new_id, int32_t left, int32_t right, StepStats &stats) {
    auto &nodes = seqs[seq_id];
    Node &a = nodes[p1];
    // 增量 pair 索引包含 stale 位置；不匹配当前 pair 时跳过，不能报错。
    if (!is_alive(a) || a.next < 0) return;
    const int32_t p2 = a.next;
    Node &b = nodes[p2];
    if (!is_alive(b) || a.val != left || b.val != right) return;

    const int32_t p0 = a.prev;
    const int32_t p3 = b.next;
    remove_pair(seq_id, p1);
    if (p0 >= 0 && a.run == 1) remove_pair(seq_id, p0);
    if (p3 >= 0 && b.run == 1) remove_pair(seq_id, p2);

    if (a.run > 1 && b.run > 1) {
      Node mid;
      mid.val = new_id;
      mid.prev = p1;
      mid.next = p2;
      mid.run = 1;
      const int32_t pm = static_cast<int32_t>(nodes.size());
      validate_node_index_capacity(static_cast<uint64_t>(pm), "apply_pair_merge middle");
      nodes.push_back(mid);
      decrement_run(seq_id, p1);
      decrement_run(seq_id, p2);
      token_count += 1;
      nodes[p1].next = pm;
      nodes[p2].prev = pm;
      add_pair(seq_id, p1, true);
      add_pair(seq_id, pm, true);
    } else if (a.run > 1 && b.run == 1) {
      b.val = new_id;
      b.run = 1;
      decrement_run(seq_id, p1);
      add_pair(seq_id, p1, true);
      if (p3 >= 0) add_pair(seq_id, p2, true);
      maybe_coalesce_adjacent(seq_id, p2);
    } else if (a.run == 1 && b.run > 1) {
      a.val = new_id;
      a.run = 1;
      decrement_run(seq_id, p2);
      if (p0 >= 0) add_pair(seq_id, p0, true);
      add_pair(seq_id, p1, true);
      if (p0 >= 0) maybe_coalesce_adjacent(seq_id, p0);
    } else {
      a.val = new_id;
      a.run = 1;
      a.next = p3;
      mark_dead(b);
      if (p3 >= 0) nodes[p3].prev = p1;
      token_count -= 1;
      if (p0 >= 0) add_pair(seq_id, p0, true);
      if (p3 >= 0) add_pair(seq_id, p1, true);
      if (p0 >= 0) maybe_coalesce_adjacent(seq_id, p0);
      if (p3 >= 0) maybe_coalesce_adjacent(seq_id, p1);
    }
    stats.real_merges += 1;
  }

  /// 扫描连续同 token 区间，并精确应用 left==right 的非重叠 merge。
  void apply_self_scan_merge(uint32_t seq_id, int32_t pos_id, int32_t new_id, int32_t token, StepStats &stats) {
    // 该入口由 pair2pos 触发；索引陈旧时会落到这些正常跳过分支。
    if (seq_id >= seqs.size()) return;
    auto &nodes = seqs[seq_id];
    if (pos_id < 0 || static_cast<size_t>(pos_id) >= nodes.size()) return;
    Node &hit = nodes[pos_id];
    if (!is_alive(hit) || hit.val != token) return;

    int32_t start = pos_id;
    while (start >= 0) {
      Node &cur = nodes[start];
      const int32_t prev = cur.prev;
      if (prev < 0) break;
      Node &p = nodes[prev];
      if (!is_alive(p) || p.val != token) break;
      start = prev;
    }

    int32_t cur_pos = start;
    while (cur_pos >= 0 && static_cast<size_t>(cur_pos) < nodes.size()) {
      Node &cur = nodes[cur_pos];
      if (!is_alive(cur) || cur.val != token) break;
      if (cur.run >= 2) {
        apply_self_merge(seq_id, cur_pos, new_id, stats);
        cur_pos = nodes[cur_pos].next;
        continue;
      }
      const int32_t next_pos = cur.next;
      if (next_pos < 0) break;
      Node &next = nodes[next_pos];
      if (!is_alive(next) || next.val != token) break;
      apply_pair_merge(seq_id, cur_pos, new_id, token, token, stats);
      cur_pos = nodes[cur_pos].next;
    }
  }

  /// 从当前有效链表完整重建 pair 位置和计数，用于调试校验。
  void rebuild_pair_index() {
    pair2pos.clear();
    pair2cnt.clear();
    pair_counts_dropped = false;
    for (uint32_t sid = 0; sid < seqs.size(); ++sid) {
      auto &nodes = seqs[sid];
      for (size_t i = 0; i < nodes.size(); ++i) {
        if (!is_alive(nodes[i])) continue;
        if (nodes[i].run >= 2) add_self_pair(sid, static_cast<int32_t>(i), false);
        if (nodes[i].next >= 0) add_pair(sid, static_cast<int32_t>(i), false);
      }
    }
  }

  /// 检查增量维护的 pair 计数是否与完整重建结果一致。
  bool debug_counts_match() {
    flat_hash_map<uint64_t, int64_t> saved = pair2cnt;
    rebuild_pair_index();
    bool ok = true;
    for (const auto &item : saved) {
      auto it = pair2cnt.find(item.first);
      int64_t rebuilt = it == pair2cnt.end() ? 0 : it->second;
      if (rebuilt != item.second) {
        ok = false;
        break;
      }
    }
    if (ok) {
      for (const auto &item : pair2cnt) {
        auto it = saved.find(item.first);
        int64_t old = it == saved.end() ? 0 : it->second;
        if (old != item.second) {
          ok = false;
          break;
        }
      }
    }
    if (!ok) {
      std::cerr << "pair count mismatch saved_size=" << saved.size()
                << " rebuilt_size=" << pair2cnt.size() << std::endl;
      int printed = 0;
      for (const auto &item : saved) {
        auto it = pair2cnt.find(item.first);
        int64_t rebuilt = it == pair2cnt.end() ? 0 : it->second;
        if (rebuilt != item.second && printed++ < 20) {
          std::cerr << " saved key=(" << unpack_left(item.first) << "," << unpack_right(item.first)
                    << ") saved=" << item.second << " rebuilt=" << rebuilt << std::endl;
        }
      }
      for (const auto &item : pair2cnt) {
        auto it = saved.find(item.first);
        int64_t old = it == saved.end() ? 0 : it->second;
        if (old != item.second && printed++ < 40) {
          std::cerr << " rebuilt key=(" << unpack_left(item.first) << "," << unpack_right(item.first)
                    << ") saved=" << old << " rebuilt=" << item.second << std::endl;
        }
      }
    }
    pair2cnt.swap(saved);
    return ok;
  }

  /// 在当前 shard 上应用一次已选中的 BPE merge，并返回本地统计信息。
  StepStats apply_merge(uint64_t key, int32_t new_id, int32_t step, bool sort_positions) {
    StepStats stats;
    stats.step = step;
    stats.left = unpack_left(key);
    stats.right = unpack_right(key);
    stats.new_id = new_id;
    auto start = Clock::now();
    delta.clear();

    auto it = pair2pos.find(key);
    if (it == pair2pos.end()) {
      // 本 shard 没有该 pair，或者该 pair 只存在于其他 rank/shard。
      stats.apply_sec = seconds_since(start);
      return stats;
    }
    std::vector<Position> positions = std::move(it->second);
    pair2pos.erase(it);
    const int32_t left = unpack_left(key);
    const int32_t right = unpack_right(key);
    if (sort_positions && left == right) {
      auto sort_start = Clock::now();
      std::sort(positions.begin(), positions.end(), [](const Position &a, const Position &b) {
        if (a.seq_id != b.seq_id) return a.seq_id < b.seq_id;
        return a.pos_id < b.pos_id;
      });
      stats.sort_sec = seconds_since(sort_start);
    }
    stats.positions_seen = positions.size();

    for (const Position &pos : positions) {
      // 位置索引是追加式维护，历史位置可能在前面轮次失效；逐条复核后跳过。
      if (pos.seq_id >= seqs.size()) continue;
      auto &nodes = seqs[pos.seq_id];
      if (pos.pos_id >= nodes.size()) continue;
      Node &a = nodes[pos.pos_id];
      if (!is_alive(a)) continue;
      if (left == right && !sort_positions) {
        apply_self_scan_merge(pos.seq_id, static_cast<int32_t>(pos.pos_id), new_id, left, stats);
      } else if (left == right && a.run >= 2) {
        if (a.val != left) continue;
        apply_self_merge(pos.seq_id, static_cast<int32_t>(pos.pos_id), new_id, stats);
      } else {
        apply_pair_merge(pos.seq_id, static_cast<int32_t>(pos.pos_id), new_id, left, right, stats);
      }
    }
    stats.changed_unique_pairs = delta.size();
    stats.tokens_after = token_count;
    stats.apply_sec = seconds_since(start);
    if (!pair_counts_dropped && std::getenv("INT_BPE_DEBUG_REBUILD")) {
      if (!debug_counts_match()) {
        throw std::runtime_error("incremental pair counts diverged after step " + std::to_string(step));
      }
    }
    return stats;
  }
};

struct ThreadPoolTrainer {
  std::vector<Shard> shards;
  std::vector<std::thread> workers;
  std::vector<StepStats> worker_stats;
  int worker_count = 1;
  uint64_t task_key = 0;
  int32_t task_new_id = 0;
  int32_t task_step = 0;
  bool task_sort_positions = false;
  size_t next_shard = 0;
  uint64_t generation = 0;
  uint64_t done = 0;
  bool stop = false;
  std::mutex mutex;
  std::condition_variable cv_start;
  std::condition_variable cv_done;

  /// 析构时停止本地工作线程，避免释放 shard 状态后仍有线程访问。
  ~ThreadPoolTrainer() {
    shutdown();
  }

  /// 唤醒所有 merge worker，请求退出并等待线程结束。
  void shutdown() {
    {
      std::lock_guard<std::mutex> lock(mutex);
      stop = true;
      generation += 1;
    }
    cv_start.notify_all();
    for (auto &worker : workers) {
      if (worker.joinable()) worker.join();
    }
  }

  /// 启动常驻本地 worker 线程，后续每轮复用它们处理 shard。
  void start_workers(int requested_workers) {
    worker_stats.resize(shards.size());
    worker_count = std::max(1, std::min<int>(requested_workers, std::max<size_t>(1, shards.size())));
    for (int wid = 0; wid < worker_count; ++wid) {
      workers.emplace_back([this]() {
        uint64_t seen = 0;
        while (true) {
          {
            std::unique_lock<std::mutex> lock(mutex);
            cv_start.wait(lock, [&]() { return stop || generation != seen; });
            if (stop) return;
            seen = generation;
          }
          while (true) {
            size_t shard_id = 0;
            {
              std::lock_guard<std::mutex> lock(mutex);
              if (next_shard >= shards.size()) break;
              shard_id = next_shard++;
            }
            StepStats stats = shards[shard_id].apply_merge(task_key, task_new_id, task_step, task_sort_positions);
            {
              std::lock_guard<std::mutex> lock(mutex);
              worker_stats[shard_id] = stats;
              done += 1;
              if (done == shards.size()) cv_done.notify_one();
            }
          }
        }
      });
    }
  }

  /// 把一次 merge 任务分发给本地 shard worker，并等待所有 shard 完成。
  std::vector<StepStats> apply(uint64_t key, int32_t new_id, int32_t step, bool sort_positions) {
    {
      std::lock_guard<std::mutex> lock(mutex);
      task_key = key;
      task_new_id = new_id;
      task_step = step;
      task_sort_positions = sort_positions;
      done = 0;
      next_shard = 0;
      generation += 1;
    }
    cv_start.notify_all();
    {
      std::unique_lock<std::mutex> lock(mutex);
      cv_done.wait(lock, [&]() { return done == shards.size(); });
    }
    return worker_stats;
  }
};

/// 按 token 数把样本切成若干份，同时保持样本边界不被切开。
std::vector<std::vector<std::vector<int32_t>>> split_for_threads(std::vector<std::vector<int32_t>> seqs, int threads) {
  threads = std::max(1, std::min<int>(threads, std::max<size_t>(1, seqs.size())));
  std::vector<uint64_t> total_prefix(seqs.size() + 1, 0);
  for (size_t i = 0; i < seqs.size(); ++i) {
    total_prefix[i + 1] = total_prefix[i] + seqs[i].size();
  }
  const uint64_t total = total_prefix.back();
  std::vector<std::vector<std::vector<int32_t>>> parts(threads);
  size_t cur = 0;
  for (int t = 0; t < threads; ++t) {
    const uint64_t target_end = total * static_cast<uint64_t>(t + 1) / static_cast<uint64_t>(threads);
    size_t end = cur;
    while (end < seqs.size() && (t + 1 == threads || total_prefix[end + 1] <= target_end)) {
      end += 1;
    }
    if (end == cur && cur < seqs.size()) end += 1;
    parts[t].reserve(end - cur);
    for (; cur < end && cur < seqs.size(); ++cur) {
      parts[t].push_back(std::move(seqs[cur]));
    }
  }
  return parts;
}

std::vector<std::vector<std::string>> split_domains_like_sequences(
    std::vector<std::string> domains,
    const std::vector<std::vector<std::vector<int32_t>>> &split) {
  std::vector<std::vector<std::string>> parts(split.size());
  size_t cur = 0;
  for (size_t t = 0; t < split.size(); ++t) {
    const size_t n = split[t].size();
    if (cur + n > domains.size()) {
      throw std::runtime_error("domain count is smaller than sequence split");
    }
    parts[t].reserve(n);
    for (size_t i = 0; i < n; ++i) {
      parts[t].push_back(std::move(domains[cur + i]));
    }
    cur += n;
  }
  if (cur != domains.size()) {
    throw std::runtime_error("domain count does not match sequence split");
  }
  return parts;
}

std::vector<std::vector<uint64_t>> split_lengths_like_sequences(
    std::vector<uint64_t> lengths,
    const std::vector<std::vector<std::vector<int32_t>>> &split) {
  std::vector<std::vector<uint64_t>> parts(split.size());
  size_t cur = 0;
  for (size_t t = 0; t < split.size(); ++t) {
    const size_t n = split[t].size();
    if (cur + n > lengths.size()) {
      throw std::runtime_error("length count is smaller than sequence split");
    }
    parts[t].reserve(n);
    for (size_t i = 0; i < n; ++i) {
      parts[t].push_back(lengths[cur + i]);
    }
    cur += n;
  }
  if (cur != lengths.size()) {
    throw std::runtime_error("length count does not match sequence split");
  }
  return parts;
}

std::vector<std::string> load_partition_domains(const std::string &path, int rank, int world_size) {
  auto runs = read_domain_runs(path);
  std::vector<std::string> domains;
  uint64_t seq_index = 0;
  for (const auto &run : runs) {
    for (uint64_t i = 0; i < run.records; ++i, ++seq_index) {
      if (world_size <= 1 || static_cast<int>(seq_index % static_cast<uint64_t>(world_size)) == rank) {
        domains.push_back(run.name);
      }
    }
  }
  return domains;
}

struct BuildTiming {
  double wall_sec = 0.0;
  double worker_sum_sec = 0.0;
  double worker_max_sec = 0.0;
  double node_sum_sec = 0.0;
  double node_max_sec = 0.0;
  double index_sum_sec = 0.0;
  double index_max_sec = 0.0;
};

/// 并行构建所有 trainer shard，并收集构建阶段耗时。
BuildTiming build_trainer_shards(ThreadPoolTrainer &trainer,
                                 std::vector<std::vector<std::vector<int32_t>>> &split,
                                 int build_workers,
                                 uint64_t reserve_pairs,
                                 bool use_rle,
                                 const std::string &build_mode,
                                 int32_t base_vocab_size) {
  auto wall_start = Clock::now();
  trainer.shards.resize(split.size());
  std::atomic<size_t> next{0};
  const int workers = std::max(1, std::min<int>(build_workers, std::max<size_t>(1, split.size())));
  std::vector<double> shard_secs(split.size(), 0.0);
  std::vector<std::thread> build_threads;
  build_threads.reserve(workers);
  for (int i = 0; i < workers; ++i) {
    build_threads.emplace_back([&]() {
      while (true) {
        const size_t shard_id = next.fetch_add(1);
        if (shard_id >= split.size()) break;
        auto shard_start = Clock::now();
        trainer.shards[shard_id].build_from_sequences(
            std::move(split[shard_id]),
            reserve_pairs / std::max<size_t>(1, split.size()) + 1024,
            use_rle,
            build_mode,
            base_vocab_size);
        shard_secs[shard_id] = seconds_since(shard_start);
      }
    });
  }
  for (auto &t : build_threads) t.join();
  BuildTiming timing;
  timing.wall_sec = seconds_since(wall_start);
  for (double sec : shard_secs) {
    timing.worker_sum_sec += sec;
    timing.worker_max_sec = std::max(timing.worker_max_sec, sec);
  }
  for (const auto &shard : trainer.shards) {
    timing.node_sum_sec += shard.build_node_sec;
    timing.node_max_sec = std::max(timing.node_max_sec, shard.build_node_sec);
    timing.index_sum_sec += shard.build_index_sec;
    timing.index_max_sec = std::max(timing.index_max_sec, shard.build_index_sec);
  }
  return timing;
}

/// 原子写出非零 pair 计数，用于文件同步版分布式训练。
void write_counts_file(const std::string &path, const flat_hash_map<uint64_t, int64_t> &counts) {
  const std::string tmp = path + ".tmp." + std::to_string(::getpid());
  std::ofstream out(tmp, std::ios::binary);
  if (!out) throw std::runtime_error("failed to open counts output: " + tmp);
  out.write(COUNTS_MAGIC, 8);
  uint64_t n = 0;
  for (const auto &item : counts) {
    if (item.second != 0) n += 1;
  }
  write_raw(out, n);
  for (const auto &item : counts) {
    if (item.second == 0) continue;
    write_raw(out, item.first);
    write_raw(out, item.second);
  }
  out.close();
  if (std::rename(tmp.c_str(), path.c_str()) != 0) {
    throw std::runtime_error("failed to rename counts output");
  }
}

/// 读取 write_counts_file() 写出的二进制 pair 计数文件。
flat_hash_map<uint64_t, int64_t> read_counts_file(const std::string &path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("failed to open counts file: " + path);
  char magic[8];
  in.read(magic, 8);
  if (!in || !std::equal(magic, magic + 8, COUNTS_MAGIC)) {
    throw std::runtime_error("invalid counts file magic: " + path);
  }
  uint64_t n = 0;
  read_raw(in, n);
  flat_hash_map<uint64_t, int64_t> counts;
  counts.reserve(n + 1);
  for (uint64_t i = 0; i < n; ++i) {
    uint64_t key = 0;
    int64_t value = 0;
    read_raw(in, key);
    read_raw(in, value);
    counts[key] += value;
  }
  return counts;
}

struct TcpStats {
  uint64_t sent = 0;
  uint64_t received = 0;
};

/// 将当前 errno 转成人类可读错误信息，并加上调用方上下文。
std::string errno_message(const std::string &prefix) {
  return prefix + ": " + std::strerror(errno);
}

/// 通过 socket 精确发送指定字节数，并处理被信号打断的系统调用。
void write_all(int fd, const void *data, size_t size, TcpStats &stats) {
  const char *ptr = static_cast<const char *>(data);
  size_t done = 0;
  while (done < size) {
    ssize_t n = ::send(fd, ptr + done, size - done, MSG_NOSIGNAL);
    if (n < 0) {
      if (errno == EINTR) continue;
      throw std::runtime_error(errno_message("socket send failed"));
    }
    if (n == 0) throw std::runtime_error("socket send returned zero");
    done += static_cast<size_t>(n);
  }
  stats.sent += size;
}

/// 从 socket 精确接收指定字节数；连接提前关闭时直接报错。
void read_all(int fd, void *data, size_t size, TcpStats &stats) {
  char *ptr = static_cast<char *>(data);
  size_t done = 0;
  while (done < size) {
    ssize_t n = ::recv(fd, ptr + done, size - done, MSG_WAITALL);
    if (n < 0) {
      if (errno == EINTR) continue;
      throw std::runtime_error(errno_message("socket recv failed"));
    }
    if (n == 0) throw std::runtime_error("socket closed while reading");
    done += static_cast<size_t>(n);
  }
  stats.received += size;
}

template <typename T>
/// 通过 TCP 控制/数据通道发送一个普通二进制值。
void send_value(int fd, const T &value, TcpStats &stats) {
  write_all(fd, &value, sizeof(T), stats);
}

template <typename T>
/// 从 TCP 控制/数据通道接收一个普通二进制值。
void recv_value(int fd, T &value, TcpStats &stats) {
  read_all(fd, &value, sizeof(T), stats);
}

/// 关闭 Nagle 算法，降低小控制消息的 TCP 延迟。
void set_tcp_no_delay(int fd) {
  int flag = 1;
  (void)::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &flag, sizeof(flag));
}

/// 为 rank0 创建 TCP 监听 socket，并返回实际绑定端口。
int create_listener(int requested_port, int &actual_port) {
  int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) throw std::runtime_error(errno_message("socket create failed"));
  int yes = 1;
  if (::setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes)) != 0) {
    ::close(fd);
    throw std::runtime_error(errno_message("setsockopt SO_REUSEADDR failed"));
  }
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = htonl(INADDR_ANY);
  addr.sin_port = htons(static_cast<uint16_t>(requested_port));
  if (::bind(fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) != 0) {
    ::close(fd);
    throw std::runtime_error(errno_message("bind failed"));
  }
  if (::listen(fd, 256) != 0) {
    ::close(fd);
    throw std::runtime_error(errno_message("listen failed"));
  }
  sockaddr_in bound{};
  socklen_t len = sizeof(bound);
  if (::getsockname(fd, reinterpret_cast<sockaddr *>(&bound), &len) != 0) {
    ::close(fd);
    throw std::runtime_error(errno_message("getsockname failed"));
  }
  actual_port = ntohs(bound.sin_port);
  return fd;
}

/// 选择一个非 loopback IPv4 地址，供其他 worker rank 连接。
std::string get_preferred_ip() {
  ifaddrs *ifaddr = nullptr;
  if (::getifaddrs(&ifaddr) != 0) {
    throw std::runtime_error(errno_message("getifaddrs failed"));
  }
  std::string fallback;
  for (ifaddrs *ifa = ifaddr; ifa != nullptr; ifa = ifa->ifa_next) {
    if (!ifa->ifa_addr || ifa->ifa_addr->sa_family != AF_INET) continue;
    if ((ifa->ifa_flags & IFF_LOOPBACK) != 0) continue;
    char host[NI_MAXHOST];
    int rc = ::getnameinfo(ifa->ifa_addr, sizeof(sockaddr_in), host, sizeof(host), nullptr, 0, NI_NUMERICHOST);
    if (rc != 0) continue;
    std::string ip(host);
    if (fallback.empty()) fallback = ip;
    if (ip.rfind("10.", 0) == 0 || ip.rfind("100.", 0) == 0 || ip.rfind("172.", 0) == 0 || ip.rfind("192.168.", 0) == 0) {
      ::freeifaddrs(ifaddr);
      return ip;
    }
  }
  ::freeifaddrs(ifaddr);
  if (!fallback.empty()) return fallback;
  return "127.0.0.1";
}

/// 将 rank0 的 host/port 通过原子写文件方式发布给其他 rank。
void write_rendezvous(const std::string &path, const std::string &host, int port) {
  const std::string tmp = path + ".tmp." + std::to_string(::getpid());
  std::ofstream out(tmp);
  if (!out) throw std::runtime_error("failed to open rendezvous output: " + tmp);
  out << host << " " << port << "\n";
  out.close();
  if (std::rename(tmp.c_str(), path.c_str()) != 0) {
    throw std::runtime_error(errno_message("failed to rename rendezvous output"));
  }
}

/// 从 rendezvous 文件读取 rank0 的 TCP 地址。
std::pair<std::string, int> read_rendezvous(const std::string &path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("failed to open rendezvous file: " + path);
  std::string host;
  int port = 0;
  in >> host >> port;
  if (host.empty() || port <= 0) throw std::runtime_error("invalid rendezvous file: " + path);
  return {host, port};
}

/// 连接 rank0；如果端点暂时不可用，则在超时前持续重试。
int connect_with_retry(const std::string &host, int port, int timeout_sec, double &wait_accum) {
  auto start = Clock::now();
  while (true) {
    int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) throw std::runtime_error(errno_message("socket create failed"));
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<uint16_t>(port));
    if (::inet_pton(AF_INET, host.c_str(), &addr.sin_addr) != 1) {
      ::close(fd);
      throw std::runtime_error("invalid IPv4 rendezvous host: " + host);
    }
    if (::connect(fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) == 0) {
      set_tcp_no_delay(fd);
      wait_accum += seconds_since(start);
      return fd;
    }
    ::close(fd);
    if (seconds_since(start) > timeout_sec) {
      throw std::runtime_error(errno_message("timed out connecting to " + host + ":" + std::to_string(port)));
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
}

/// 通过 TCP 分块发送非零 pair 计数，避免一次性构造过大的网络包。
void send_counts_socket(int fd, const flat_hash_map<uint64_t, int64_t> &counts, TcpStats &stats) {
  uint64_t n = 0;
  for (const auto &item : counts) {
    if (item.second != 0) n += 1;
  }
  send_value(fd, n, stats);
  std::vector<CountWire> buffer;
  buffer.reserve(static_cast<size_t>(std::min<uint64_t>(n, kCountWireChunkRecords)));
  for (const auto &item : counts) {
    if (item.second == 0) continue;
    buffer.push_back({item.first, item.second});
    if (buffer.size() == kCountWireChunkRecords) {
      write_all(fd, buffer.data(), buffer.size() * sizeof(CountWire), stats);
      buffer.clear();
    }
  }
  if (!buffer.empty()) {
    write_all(fd, buffer.data(), buffer.size() * sizeof(CountWire), stats);
  }
}

/// 发送一个已经知道条目数的计数流；emit 负责逐条调用内部发送函数。
template <typename EmitFn>
void send_count_stream_socket(int fd, uint64_t n, TcpStats &stats, EmitFn &&emit) {
  send_value(fd, n, stats);
  std::vector<CountWire> buffer;
  buffer.reserve(static_cast<size_t>(std::min<uint64_t>(n, kCountWireChunkRecords)));
  auto push = [&](uint64_t key, int64_t value) {
    if (value == 0) return;
    buffer.push_back({key, value});
    if (buffer.size() == kCountWireChunkRecords) {
      write_all(fd, buffer.data(), buffer.size() * sizeof(CountWire), stats);
      buffer.clear();
    }
  };
  emit(push);
  if (!buffer.empty()) {
    write_all(fd, buffer.data(), buffer.size() * sizeof(CountWire), stats);
  }
}

/// 按 shard 流式发送初始 pair 计数，不额外合并出一张本地大 map。
void send_initial_counts_socket_from_shards(int fd, const std::vector<Shard> &shards, TcpStats &stats) {
  uint64_t n = 0;
  for (const auto &shard : shards) {
    shard.for_each_pair_count([&](uint64_t, int64_t) { n += 1; });
  }
  send_count_stream_socket(fd, n, stats, [&](auto &&push) {
    for (const auto &shard : shards) {
      shard.for_each_pair_count(push);
    }
  });
}

/// 按 shard 流式发送本轮 delta，不额外合并出一张本地大 map。
void send_delta_counts_socket_from_shards(int fd, const std::vector<Shard> &shards, TcpStats &stats) {
  uint64_t n = 0;
  for (const auto &shard : shards) {
    shard.for_each_delta([&](uint64_t, int64_t) { n += 1; });
  }
  send_count_stream_socket(fd, n, stats, [&](auto &&push) {
    for (const auto &shard : shards) {
      shard.for_each_delta(push);
    }
  });
}

template <typename Fn>
/// 分块接收 pair 计数，并对每个非零条目调用 fn(key, value)。
void recv_counts_socket_for_each(int fd, TcpStats &stats, Fn &&fn) {
  uint64_t n = 0;
  recv_value(fd, n, stats);
  std::vector<CountWire> buffer;
  buffer.resize(static_cast<size_t>(std::min<uint64_t>(n, kCountWireChunkRecords)));
  uint64_t done = 0;
  while (done < n) {
    const size_t chunk = static_cast<size_t>(std::min<uint64_t>(n - done, kCountWireChunkRecords));
    if (buffer.size() < chunk) buffer.resize(chunk);
    read_all(fd, buffer.data(), chunk * sizeof(CountWire), stats);
    for (size_t i = 0; i < chunk; ++i) {
      if (buffer[i].value != 0) fn(buffer[i].key, buffer[i].value);
    }
    done += chunk;
  }
}

struct MergeWire {
  uint8_t stop = 0;
  int32_t step = 0;
  uint64_t key = 0;
  int32_t new_id = 0;
  int64_t freq = -1;
};

/// 通过 TCP 发送某一轮选中的 merge 指令。
void send_merge_socket(int fd, const MergeWire &msg, TcpStats &stats) {
  send_value(fd, msg.stop, stats);
  send_value(fd, msg.step, stats);
  send_value(fd, msg.key, stats);
  send_value(fd, msg.new_id, stats);
  send_value(fd, msg.freq, stats);
}

/// 通过 TCP 接收某一轮选中的 merge 指令。
MergeWire recv_merge_socket(int fd, TcpStats &stats) {
  MergeWire msg;
  recv_value(fd, msg.stop, stats);
  recv_value(fd, msg.step, stats);
  recv_value(fd, msg.key, stats);
  recv_value(fd, msg.new_id, stats);
  recv_value(fd, msg.freq, stats);
  return msg;
}

/// 训练循环开始前的 TCP barrier，rank0 侧等待所有 worker 就绪。
void tcp_barrier_rank0(const std::vector<int> &peer_fds, int world_size, TcpStats &stats) {
  uint8_t ready = 0;
  for (int r = 1; r < world_size; ++r) {
    recv_value(peer_fds[r], ready, stats);
    if (ready != 1) throw std::runtime_error("invalid tcp barrier ready byte");
  }
  uint8_t go = 2;
  for (int r = 1; r < world_size; ++r) {
    send_value(peer_fds[r], go, stats);
  }
}

/// 训练循环开始前的 TCP barrier，worker 侧通知 rank0 并等待放行。
void tcp_barrier_worker(int server_fd, TcpStats &stats) {
  uint8_t ready = 1;
  send_value(server_fd, ready, stats);
  uint8_t go = 0;
  recv_value(server_fd, go, stats);
  if (go != 2) throw std::runtime_error("invalid tcp barrier release byte");
}

/// 为文件同步版分布式模式原子写出一轮 merge 指令。
void write_merge_file(const std::string &path, bool stop, int32_t step, uint64_t key, int32_t new_id, int64_t freq) {
  const std::string tmp = path + ".tmp." + std::to_string(::getpid());
  std::ofstream out(tmp, std::ios::binary);
  if (!out) throw std::runtime_error("failed to open merge output: " + tmp);
  out.write(MERGE_MAGIC, 8);
  uint8_t stop_u8 = stop ? 1 : 0;
  write_raw(out, stop_u8);
  write_raw(out, step);
  write_raw(out, key);
  write_raw(out, new_id);
  write_raw(out, freq);
  out.close();
  if (std::rename(tmp.c_str(), path.c_str()) != 0) {
    throw std::runtime_error("failed to rename merge output");
  }
}

/// 读取 write_merge_file() 写出的一轮 merge 指令。
void read_merge_file(const std::string &path, bool &stop, int32_t &step, uint64_t &key, int32_t &new_id, int64_t &freq) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("failed to open merge file: " + path);
  char magic[8];
  in.read(magic, 8);
  if (!in || !std::equal(magic, magic + 8, MERGE_MAGIC)) {
    throw std::runtime_error("invalid merge file magic: " + path);
  }
  uint8_t stop_u8 = 0;
  read_raw(in, stop_u8);
  read_raw(in, step);
  read_raw(in, key);
  read_raw(in, new_id);
  read_raw(in, freq);
  stop = stop_u8 != 0;
}

/// 将一个计数 delta 应用到全局 pair 计数，并按需刷新候选优先队列。
void apply_delta(flat_hash_map<uint64_t, int64_t> &global_counts,
                 std::priority_queue<Candidate, std::vector<Candidate>, CandidateLess> &pq,
                 uint64_t key, int64_t delta,
                 int64_t min_frequency) {
  if (delta == 0) return;
  int64_t next = 0;
  auto inserted = global_counts.emplace(key, delta);
  if (inserted.second) {
    next = delta;
  } else {
    next = inserted.first->second + delta;
    inserted.first->second = next;
  }
  if (next >= min_frequency) pq.push({key, next});
}

/// 跳过优先队列中的过期候选，直到找到当前真实最高频 pair。
Candidate select_best(flat_hash_map<uint64_t, int64_t> &global_counts,
                      std::priority_queue<Candidate, std::vector<Candidate>, CandidateLess> &pq,
                      int64_t min_frequency) {
  while (!pq.empty()) {
    Candidate cand = pq.top();
    auto it = global_counts.find(cand.key);
    int64_t real = it == global_counts.end() ? 0 : it->second;
    if (real != cand.count) {
      pq.pop();
      if (real >= min_frequency) pq.push({cand.key, real});
      continue;
    }
    if (real < min_frequency) return {0, -1};
    return cand;
  }
  return {0, -1};
}

/// 对当前 JSON 写出器需要处理的少量特殊字符做转义。
std::string json_escape(const std::string &s) {
  std::ostringstream out;
  for (char c : s) {
    if (c == '"' || c == '\\') out << '\\' << c;
    else if (c == '\n') out << "\\n";
    else out << c;
  }
  return out.str();
}

std::vector<DomainStat> read_domain_runs(const std::string &path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("failed to open domain runs: " + path);
  std::vector<DomainStat> runs;
  std::string line;
  int line_no = 0;
  while (std::getline(in, line)) {
    line_no += 1;
    if (line.empty()) continue;
    std::stringstream ss(line);
    std::string domain;
    std::string records_text;
    if (!std::getline(ss, domain, '\t') || !std::getline(ss, records_text, '\t')) {
      throw std::runtime_error("domain runs must be TSV: " + path + ":" + std::to_string(line_no));
    }
    if (domain.empty()) {
      throw std::runtime_error("empty domain in domain runs: " + path + ":" + std::to_string(line_no));
    }
    DomainStat run;
    run.name = domain;
    run.records = static_cast<uint64_t>(std::stoull(records_text));
    if (run.records == 0) {
      throw std::runtime_error("zero-length domain run in " + path + ":" + std::to_string(line_no));
    }
    runs.push_back(std::move(run));
  }
  if (runs.empty()) throw std::runtime_error("empty domain runs: " + path);
  return runs;
}

void write_domain_summary(const std::string &path,
                          const Args &args,
                          const std::vector<DomainStat> &stats,
                          uint64_t records,
                          uint64_t input_tokens,
                          uint64_t output_tokens,
                          uint64_t rules_loaded,
                          double total_sec) {
  std::ofstream out(path);
  if (!out) throw std::runtime_error("failed to write domain summary: " + path);
  out << "{\n"
      << "  \"mode\": \"apply_merges_domain_stats\",\n"
      << "  \"input\": \"" << json_escape(args.input) << "\",\n"
      << "  \"apply_merges\": \"" << json_escape(args.apply_merges) << "\",\n"
      << "  \"domain_runs\": \"" << json_escape(args.domain_runs) << "\",\n"
      << "  \"rank\": " << args.rank << ",\n"
      << "  \"load_rank\": " << args.load_rank << ",\n"
      << "  \"load_world_size\": " << args.load_world_size << ",\n"
      << "  \"threads\": " << args.threads << ",\n"
      << "  \"data_shards\": " << (args.data_shards > 0 ? args.data_shards : args.threads) << ",\n"
      << "  \"rules_loaded\": " << rules_loaded << ",\n"
      << "  \"records\": " << records << ",\n"
      << "  \"input_tokens\": " << input_tokens << ",\n"
      << "  \"output_tokens\": " << output_tokens << ",\n"
      << "  \"compression_ratio\": "
      << (output_tokens > 0 ? static_cast<double>(input_tokens) / static_cast<double>(output_tokens) : 0.0) << ",\n"
      << "  \"total_sec\": " << total_sec << ",\n"
      << "  \"domains\": {\n";
  for (size_t i = 0; i < stats.size(); ++i) {
    const auto &s = stats[i];
    out << "    \"" << json_escape(s.name) << "\": {\n"
        << "      \"records\": " << s.records << ",\n"
        << "      \"input_tokens\": " << s.input_tokens << ",\n"
        << "      \"output_tokens\": " << s.output_tokens << ",\n"
        << "      \"compression_ratio\": "
        << (s.output_tokens > 0 ? static_cast<double>(s.input_tokens) / static_cast<double>(s.output_tokens) : 0.0) << ",\n"
        << "      \"avg_input_len\": "
        << (s.records > 0 ? static_cast<double>(s.input_tokens) / static_cast<double>(s.records) : 0.0) << ",\n"
        << "      \"avg_output_len\": "
        << (s.records > 0 ? static_cast<double>(s.output_tokens) / static_cast<double>(s.records) : 0.0) << ",\n"
        << "      \"max_input_len\": " << s.max_input_len << ",\n"
        << "      \"max_output_len\": " << s.max_output_len << "\n"
        << "    }" << (i + 1 == stats.size() ? "\n" : ",\n");
  }
  out << "  }\n"
      << "}\n";
}

/// 写出本次运行的 summary JSON，包括配置、规模、耗时和 IO 统计。
void write_summary(const std::string &path, const Summary &summary) {
  std::ofstream out(path);
  if (!out) throw std::runtime_error("failed to write summary: " + path);
  out << "{\n";
  out << "  \"mode\": \"" << json_escape(summary.mode) << "\",\n";
  out << "  \"rank\": " << summary.rank << ",\n";
  out << "  \"world_size\": " << summary.world_size << ",\n";
  out << "  \"threads\": " << summary.threads << ",\n";
  out << "  \"data_shards\": " << summary.data_shards << ",\n";
  out << "  \"input_shard\": " << (summary.input_shard ? "true" : "false") << ",\n";
  out << "  \"load_rank\": " << summary.load_rank << ",\n";
  out << "  \"load_world_size\": " << summary.load_world_size << ",\n";
  out << "  \"num_merges_requested\": " << summary.num_merges_requested << ",\n";
  out << "  \"num_merges_performed\": " << summary.num_merges_performed << ",\n";
  out << "  \"min_frequency\": " << summary.min_frequency << ",\n";
  out << "  \"use_rle\": " << (summary.use_rle ? "true" : "false") << ",\n";
  out << "  \"sort_positions\": " << (summary.sort_positions ? "true" : "false") << ",\n";
  out << "  \"build_mode\": \"" << json_escape(summary.build_mode) << "\",\n";
  out << "  \"pq_min_frequency_filter\": " << (summary.pq_min_frequency_filter ? "true" : "false") << ",\n";
  out << "  \"count_wire_chunk_records\": " << summary.count_wire_chunk_records << ",\n";
  out << "  \"node_size_bytes\": " << summary.node_size_bytes << ",\n";
  out << "  \"position_size_bytes\": " << summary.position_size_bytes << ",\n";
  out << "  \"reserve_pairs\": " << summary.reserve_pairs << ",\n";
  out << "  \"global_reserve_pairs\": " << summary.global_reserve_pairs << ",\n";
  out << "  \"checkpoint_every\": " << summary.checkpoint_every << ",\n";
  out << "  \"resume_merges\": \"" << json_escape(summary.resume_merges) << "\",\n";
  out << "  \"resume_steps\": " << summary.resume_steps << ",\n";
  out << "  \"drop_local_counts_after_init\": " << (summary.drop_local_counts_after_init ? "true" : "false") << ",\n";
  out << "  \"stream_local_counts\": " << (summary.stream_local_counts ? "true" : "false") << ",\n";
  out << "  \"stream_local_deltas\": " << (summary.stream_local_deltas ? "true" : "false") << ",\n";
  out << "  \"input_sequences\": " << summary.input_sequences << ",\n";
  out << "  \"input_tokens\": " << summary.input_tokens << ",\n";
  out << "  \"final_tokens\": " << summary.final_tokens << ",\n";
  out << "  \"encoded_output\": \"" << json_escape(summary.encoded_output) << "\",\n";
  out << "  \"encoded_compact_int\": " << (summary.encoded_compact_int ? "true" : "false") << ",\n";
  out << "  \"encoded_records\": " << summary.encoded_records << ",\n";
  out << "  \"encoded_tokens\": " << summary.encoded_tokens << ",\n";
  out << "  \"encoded_min_token\": " << summary.encoded_min_token << ",\n";
  out << "  \"encoded_max_token\": " << summary.encoded_max_token << ",\n";
  out << "  \"token_offset\": " << summary.token_offset << ",\n";
  out << "  \"add_eos\": " << (summary.add_eos ? "true" : "false") << ",\n";
  out << "  \"eos_token_id\": " << summary.eos_token_id << ",\n";
  out << "  \"initial_unique_pairs\": " << summary.initial_unique_pairs << ",\n";
  out << "  \"final_unique_pairs\": " << summary.final_unique_pairs << ",\n";
  out << "  \"apply_positions_seen\": " << summary.apply_positions_seen << ",\n";
  out << "  \"apply_real_merges\": " << summary.apply_real_merges << ",\n";
  out << std::fixed << std::setprecision(6);
  out << "  \"connect_sec\": " << summary.connect_sec << ",\n";
  out << "  \"load_sec\": " << summary.load_sec << ",\n";
  out << "  \"split_sec\": " << summary.split_sec << ",\n";
  out << "  \"build_sec\": " << summary.build_sec << ",\n";
  out << "  \"build_worker_sum_sec\": " << summary.build_worker_sum_sec << ",\n";
  out << "  \"build_worker_max_sec\": " << summary.build_worker_max_sec << ",\n";
  out << "  \"build_node_sum_sec\": " << summary.build_node_sum_sec << ",\n";
  out << "  \"build_node_max_sec\": " << summary.build_node_max_sec << ",\n";
  out << "  \"build_index_sum_sec\": " << summary.build_index_sum_sec << ",\n";
  out << "  \"build_index_max_sec\": " << summary.build_index_max_sec << ",\n";
  out << "  \"local_initial_sec\": " << summary.local_initial_sec << ",\n";
  out << "  \"initial_reduce_sec\": " << summary.initial_reduce_sec << ",\n";
  out << "  \"barrier_sec\": " << summary.barrier_sec << ",\n";
  out << "  \"select_sec\": " << summary.select_sec << ",\n";
  out << "  \"apply_sec\": " << summary.apply_sec << ",\n";
  out << "  \"sort_sec\": " << summary.sort_sec << ",\n";
  out << "  \"delta_reduce_sec\": " << summary.delta_reduce_sec << ",\n";
  out << "  \"sync_wait_sec\": " << summary.sync_wait_sec << ",\n";
  out << "  \"write_sec\": " << summary.write_sec << ",\n";
  out << "  \"train_sec\": " << summary.train_sec << ",\n";
  out << "  \"network_bytes_sent\": " << summary.network_bytes_sent << ",\n";
  out << "  \"network_bytes_recv\": " << summary.network_bytes_recv << ",\n";
  out << "  \"total_sec\": " << summary.total_sec << "\n";
  out << "}\n";
}

/// 将学到的 BPE merge 表写成 CSV。
void write_merges(const std::string &path, const Summary &summary) {
  std::ofstream out(path);
  if (!out) throw std::runtime_error("failed to write merges: " + path);
  out << "step,left,right,new_id,freq,real_merges,positions_seen\n";
  for (const auto &m : summary.merges) {
    out << m.step << ","
        << m.left << ","
        << m.right << ","
        << m.new_id << ","
        << m.freq << ","
        << m.real_merges << ","
        << m.positions_seen << "\n";
  }
}

/// 原子写出 merge CSV，用作可恢复训练的 checkpoint。
void write_merges_atomic(const std::string &path, const Summary &summary) {
  const std::string tmp = path + ".tmp." + std::to_string(::getpid());
  write_merges(tmp, summary);
  atomic_rename(tmp, path);
}

/// 将每轮耗时、计数和通信诊断信息写成 CSV。
void write_profile(const std::string &path, const Summary &summary) {
  std::ofstream out(path);
  if (!out) throw std::runtime_error("failed to write profile: " + path);
  out << "step,left,right,new_id,freq,real_merges,positions_seen,"
      << "shard_delta_unique_sum,local_delta_unique,global_unique,"
      << "select_sec,merge_sync_sec,apply_wall_sec,worker_apply_sum_sec,worker_apply_max_sec,sort_sum_sec,"
      << "local_delta_build_sec,delta_send_sec,delta_local_apply_sec,delta_recv_apply_sec,delta_reduce_sec,step_wall_sec\n";
  out << std::fixed << std::setprecision(9);
  for (const auto &p : summary.profiles) {
    out << p.step << ","
        << p.left << ","
        << p.right << ","
        << p.new_id << ","
        << p.freq << ","
        << p.real_merges << ","
        << p.positions_seen << ","
        << p.shard_delta_unique_sum << ","
        << p.local_delta_unique << ","
        << p.global_unique << ","
        << p.select_sec << ","
        << p.merge_sync_sec << ","
        << p.apply_wall_sec << ","
        << p.worker_apply_sum_sec << ","
        << p.worker_apply_max_sec << ","
        << p.sort_sum_sec << ","
        << p.local_delta_build_sec << ","
        << p.delta_send_sec << ","
        << p.delta_local_apply_sec << ","
        << p.delta_recv_apply_sec << ","
        << p.delta_reduce_sec << ","
        << p.step_wall_sec << "\n";
  }
}

/// 原子写出 profile CSV，用于和 merge checkpoint 对齐观察进度。
void write_profile_atomic(const std::string &path, const Summary &summary) {
  const std::string tmp = path + ".tmp." + std::to_string(::getpid());
  write_profile(tmp, summary);
  atomic_rename(tmp, path);
}

/// 写出当前 rank 的 checkpoint 文件。rank0 会额外写通用 merges/profile 文件。
void maybe_write_checkpoint(const Args &args, const Summary &summary, int32_t completed_steps) {
  if (args.checkpoint_every <= 0 || completed_steps <= 0) return;
  if (completed_steps % args.checkpoint_every != 0) return;
  const std::string ckpt_dir = join_path(args.out_dir, "checkpoints");
  ensure_dir(ckpt_dir);
  write_merges_atomic(join_path(ckpt_dir, "merges_rank_" + std::to_string(args.rank) + ".csv"), summary);
  write_profile_atomic(join_path(ckpt_dir, "profile_rank_" + std::to_string(args.rank) + ".csv"), summary);
  if (args.rank == 0) {
    write_merges_atomic(join_path(ckpt_dir, "merges.csv"), summary);
    write_profile_atomic(join_path(ckpt_dir, "profile.csv"), summary);
    const std::string marker_tmp = join_path(ckpt_dir, "latest.txt.tmp." + std::to_string(::getpid()));
    std::ofstream marker(marker_tmp);
    if (!marker) throw std::runtime_error("failed to write checkpoint marker");
    marker << completed_steps << "\n";
    marker.close();
    atomic_rename(marker_tmp, join_path(ckpt_dir, "latest.txt"));
  }
}

/// 从 CSV 一行读取一个 merge 记录；格式必须和 write_merges() 输出一致。
MergeRecord parse_merge_record_csv(const std::string &line, const std::string &path, int line_no) {
  std::stringstream ss(line);
  std::string field;
  std::vector<std::string> fields;
  while (std::getline(ss, field, ',')) fields.push_back(field);
  if (fields.size() != 7) {
    throw std::runtime_error("invalid merge csv field count in " + path + ":" + std::to_string(line_no));
  }
  MergeRecord rec;
  rec.step = std::stoi(fields[0]);
  rec.left = std::stoi(fields[1]);
  rec.right = std::stoi(fields[2]);
  rec.new_id = std::stoi(fields[3]);
  rec.freq = std::stoll(fields[4]);
  rec.real_merges = static_cast<uint64_t>(std::stoull(fields[5]));
  rec.positions_seen = static_cast<uint64_t>(std::stoull(fields[6]));
  return rec;
}

/// 校验 resume merge 行是否严格连续，new_id 是否等于 base_vocab_size + step。
void validate_resume_records(const std::vector<MergeRecord> &records,
                             const Args &args,
                             int32_t base_vocab_size,
                             const std::string &path) {
  for (size_t i = 0; i < records.size(); ++i) {
    const MergeRecord &rec = records[i];
    if (rec.step != static_cast<int32_t>(i)) {
      throw std::runtime_error("non-contiguous resume step in " + path + ": row=" + std::to_string(i));
    }
    const int32_t expected_new_id = base_vocab_size + rec.step;
    if (rec.new_id != expected_new_id) {
      throw std::runtime_error(
          "resume new_id does not match input vocabulary in " + path
          + ": step=" + std::to_string(rec.step)
          + " expected_new_id=" + std::to_string(expected_new_id)
          + " actual_new_id=" + std::to_string(rec.new_id)
          + " base_vocab_size=" + std::to_string(base_vocab_size));
    }
    if (rec.freq < args.min_frequency) {
      throw std::runtime_error(
          "resume merge frequency is below current --min-frequency in " + path
          + ": step=" + std::to_string(rec.step)
          + " freq=" + std::to_string(rec.freq)
          + " min_frequency=" + std::to_string(args.min_frequency));
    }
  }
}

/// 读取已有 merge CSV，供断点恢复时重放已完成的 merge。
std::vector<MergeRecord> read_merge_records_csv(const std::string &path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("failed to open resume merges: " + path);
  std::string line;
  if (!std::getline(in, line)) return {};
  if (line != "step,left,right,new_id,freq,real_merges,positions_seen") {
    throw std::runtime_error("invalid merge csv header: " + path);
  }
  std::vector<MergeRecord> records;
  int line_no = 1;
  while (std::getline(in, line)) {
    line_no += 1;
    if (line.empty()) continue;
    MergeRecord rec = parse_merge_record_csv(line, path, line_no);
    if (rec.step != static_cast<int32_t>(records.size())) {
      throw std::runtime_error("non-contiguous merge step in " + path + ":" + std::to_string(line_no));
    }
    records.push_back(rec);
  }
  return records;
}

/// 根据 resume CSV 确定当前 rank 应该读取哪个文件。
std::vector<MergeRecord> load_resume_merges_for_rank(const Args &args, int32_t base_vocab_size) {
  if (args.resume_merges.empty()) return {};
  std::string path = args.resume_merges;
  if (args.rank != 0) {
    const std::string suffix = ".csv";
    if (path.size() >= suffix.size() && path.compare(path.size() - suffix.size(), suffix.size(), suffix) == 0) {
      std::string rank_path;
      if (path.size() >= std::string("merges.csv").size() &&
          path.compare(path.size() - std::string("merges.csv").size(), std::string("merges.csv").size(), "merges.csv") == 0) {
        rank_path = path.substr(0, path.size() - std::string("merges.csv").size())
            + "merges_rank_" + std::to_string(args.rank) + suffix;
      } else {
        rank_path = path.substr(0, path.size() - suffix.size()) + "_rank_" + std::to_string(args.rank) + suffix;
      }
      if (file_exists(rank_path)) {
        path = rank_path;
      } else {
        throw std::runtime_error(
            "missing rank-specific resume merges: " + rank_path
            + " (derived from " + args.resume_merges + ")");
      }
    }
  }
  auto records = read_merge_records_csv(path);
  validate_resume_records(records, args, base_vocab_size, path);
  return records;
}

Summary run_apply_stats(const Args &args) {
  Summary summary;
  summary.mode = "apply_merges_domain_stats";
  summary.rank = args.rank;
  summary.world_size = args.world_size;
  summary.threads = args.threads;
  summary.data_shards = args.data_shards > 0 ? args.data_shards : args.threads;
  summary.input_shard = args.input_shard;
  auto load_partition = effective_load_partition(args);
  summary.load_rank = load_partition.first;
  summary.load_world_size = load_partition.second;
  summary.num_merges_requested = args.num_merges;
  summary.min_frequency = args.min_frequency;
  summary.use_rle = args.use_rle;
  summary.sort_positions = args.sort_positions;
  summary.build_mode = args.build_mode;
  summary.reserve_pairs = args.reserve_pairs;
  summary.global_reserve_pairs = args.global_reserve_pairs > 0 ? args.global_reserve_pairs : args.reserve_pairs;

  auto total_start = Clock::now();
  auto load_start = Clock::now();
  SampleMeta meta;
  auto seqs = load_sample(args.input, summary.load_rank, summary.load_world_size, meta);
  summary.load_sec = seconds_since(load_start);
  summary.input_sequences = seqs.size();
  std::vector<uint64_t> input_lengths;
  input_lengths.reserve(seqs.size());
  for (const auto &seq : seqs) input_lengths.push_back(seq.size());
  for (const auto &seq : seqs) summary.input_tokens += seq.size();
  auto domains = load_partition_domains(args.domain_runs, summary.load_rank, summary.load_world_size);
  if (domains.size() != seqs.size()) {
    throw std::runtime_error(
        "domain count does not match loaded sequence count: domains=" + std::to_string(domains.size())
        + " seqs=" + std::to_string(seqs.size()));
  }

  auto split_start = Clock::now();
  auto split = split_for_threads(std::move(seqs), summary.data_shards);
  auto domain_split = split_domains_like_sequences(std::move(domains), split);
  auto length_split = split_lengths_like_sequences(std::move(input_lengths), split);
  summary.split_sec = seconds_since(split_start);
  ThreadPoolTrainer trainer;
  BuildTiming build_timing = build_trainer_shards(
      trainer, split, args.threads, args.reserve_pairs, args.use_rle, args.build_mode, meta.max_token + 1);
  summary.build_sec = build_timing.wall_sec;
  summary.build_worker_sum_sec = build_timing.worker_sum_sec;
  summary.build_worker_max_sec = build_timing.worker_max_sec;
  summary.build_node_sum_sec = build_timing.node_sum_sec;
  summary.build_node_max_sec = build_timing.node_max_sec;
  summary.build_index_sum_sec = build_timing.index_sum_sec;
  summary.build_index_max_sec = build_timing.index_max_sec;
  if (args.drop_local_counts_after_init) {
    for (auto &shard : trainer.shards) shard.drop_pair_counts();
  }
  trainer.start_workers(args.threads);

  auto records = read_merge_records_csv(args.apply_merges);
  if (args.num_merges >= 0 && static_cast<int32_t>(records.size()) > args.num_merges) {
    records.resize(static_cast<size_t>(args.num_merges));
  }
  summary.num_merges_requested = static_cast<int32_t>(records.size());
  auto train_start = Clock::now();
  for (size_t i = 0; i < records.size(); ++i) {
    const auto &rec = records[i];
    auto apply_start = Clock::now();
    const uint64_t key = pack_pair(rec.left, rec.right);
    auto stats = trainer.apply(key, rec.new_id, rec.step, args.sort_positions);
    summary.apply_sec += seconds_since(apply_start);
    uint64_t real_merges = 0;
    uint64_t positions_seen = 0;
    double sort_sec = 0.0;
    for (const auto &s : stats) {
      real_merges += s.real_merges;
      positions_seen += s.positions_seen;
      sort_sec += s.sort_sec;
    }
    summary.apply_real_merges += real_merges;
    summary.apply_positions_seen += positions_seen;
    summary.sort_sec += sort_sec;
    summary.num_merges_performed += 1;
    if (i == 0 || ((i + 1) % static_cast<size_t>(std::max(1, args.log_every)) == 0)) {
      const double elapsed = seconds_since(train_start);
      std::cerr << "apply-stats progress step=" << (i + 1)
                << " real_merges=" << real_merges
                << " elapsed_sec=" << elapsed;
      write_progress_fields(std::cerr, static_cast<int32_t>(i + 1), static_cast<int32_t>(records.size()), elapsed);
      std::cerr << std::endl;
    }
  }
  for (auto &shard : trainer.shards) summary.final_tokens += shard.token_count;
  summary.train_sec = seconds_since(train_start);
  summary.total_sec = seconds_since(total_start);

  std::vector<DomainStat> domain_stats;
  flat_hash_map<std::string, size_t> domain_to_index;
  uint64_t selected_records = 0;
  uint64_t selected_input_tokens = 0;
  uint64_t selected_output_tokens = 0;
  for (size_t shard_id = 0; shard_id < trainer.shards.size(); ++shard_id) {
    const auto &shard = trainer.shards[shard_id];
    if (shard.seqs.size() != domain_split[shard_id].size()) {
      throw std::runtime_error("domain split does not match trainer shard sequence count");
    }
    if (shard.seqs.size() != length_split[shard_id].size()) {
      throw std::runtime_error("length split does not match trainer shard sequence count");
    }
    for (size_t seq_id = 0; seq_id < shard.seqs.size(); ++seq_id) {
      const auto &nodes = shard.seqs[seq_id];
      uint64_t in_len = length_split[shard_id][seq_id];
      uint64_t out_len = 0;
      for (const auto &node : nodes) {
        if (!is_alive(node)) continue;
        out_len += node.run;
      }
      // 当前链表只保存 apply 后状态；输入长度来自 header 分区加载时已经汇总。
      // 这里用 nodes 的原始槽位数作为非 RLE 输入长度；本项目 apply-stats 使用 --no-rle。
      if (args.use_rle) {
        throw std::runtime_error("domain apply stats currently expects --no-rle");
      }
      const std::string &domain = domain_split[shard_id][seq_id];
      auto inserted = domain_to_index.emplace(domain, domain_stats.size());
      if (inserted.second) {
        DomainStat stat;
        stat.name = domain;
        domain_stats.push_back(std::move(stat));
      }
      DomainStat &stat = domain_stats[inserted.first->second];
      stat.records += 1;
      stat.input_tokens += in_len;
      stat.output_tokens += out_len;
      stat.max_input_len = std::max(stat.max_input_len, in_len);
      stat.max_output_len = std::max(stat.max_output_len, out_len);
      selected_records += 1;
      selected_input_tokens += in_len;
      selected_output_tokens += out_len;
    }
  }
  write_domain_summary(
      join_path(args.out_dir, "domain_stats_rank_" + std::to_string(args.rank) + ".json"),
      args,
      domain_stats,
      selected_records,
      selected_input_tokens,
      selected_output_tokens,
      records.size(),
      summary.total_sec);
  return summary;
}

std::vector<MergeRecord> read_fixed_apply_records(const Args &args) {
  auto records = read_merge_records_csv(args.apply_merges);
  if (args.apply_skip < 0) {
    throw std::runtime_error("--apply-skip must be non-negative");
  }
  if (args.apply_skip > 0) {
    if (static_cast<size_t>(args.apply_skip) > records.size()) {
      throw std::runtime_error("--apply-skip exceeds merge record count");
    }
    records.erase(records.begin(), records.begin() + args.apply_skip);
  }
  if (args.num_merges >= 0 && static_cast<int32_t>(records.size()) > args.num_merges) {
    records.resize(static_cast<size_t>(args.num_merges));
  }
  return records;
}

void write_token_u16(std::ofstream &out, int32_t token) {
  if (token < 0 || token > std::numeric_limits<uint16_t>::max()) {
    throw std::runtime_error("encoded token does not fit uint16: " + std::to_string(token));
  }
  const uint16_t value = static_cast<uint16_t>(token);
  write_raw(out, value);
}

void write_encoded_token_stream(const Args &args, const ThreadPoolTrainer &trainer, Summary &summary) {
  if (args.encoded_output.empty()) {
    throw std::runtime_error("--encoded-output is required with --mode apply-encode");
  }
  if (args.encoded_compact_int && args.add_eos) {
    throw std::runtime_error("--encoded-compact-int requires --no-eos because compact sample shards keep original sample boundaries");
  }
  if (args.encoded_compact_int && args.token_offset != 0) {
    throw std::runtime_error("--encoded-compact-int requires --token-offset 0 so the output can be read as an int BPE sample shard");
  }
  if (file_exists(args.encoded_output) && !args.overwrite_output) {
    throw std::runtime_error("encoded output exists; pass --overwrite-output to replace: " + args.encoded_output);
  }
  const auto start = Clock::now();
  const std::string tmp_output = args.encoded_output + ".tmp";
  std::ofstream out(tmp_output, std::ios::binary);
  if (!out) {
    throw std::runtime_error("failed to open encoded output tmp: " + tmp_output);
  }

  summary.encoded_output = args.encoded_output;
  summary.token_offset = args.token_offset;
  summary.eos_token_id = args.eos_token_id;
  summary.add_eos = args.add_eos;
  summary.encoded_compact_int = args.encoded_compact_int;
  summary.encoded_records = 0;
  summary.encoded_tokens = 0;
  summary.encoded_min_token = std::numeric_limits<int32_t>::max();
  summary.encoded_max_token = -1;

  if (args.encoded_compact_int) {
    write_sample_header(out, 0, 0, -1);
    std::vector<int32_t> buffer;
    for (const auto &shard : trainer.shards) {
      for (const auto &nodes : shard.seqs) {
        uint64_t alive_len64 = 0;
        for (const auto &node : nodes) {
          if (!is_alive(node)) continue;
          alive_len64 += node.run;
        }
        if (alive_len64 > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
          throw std::runtime_error(
              "encoded sequence length exceeds uint32: " + std::to_string(alive_len64));
        }
        const uint32_t alive_len = static_cast<uint32_t>(alive_len64);
        write_raw(out, alive_len);
        buffer.clear();
        buffer.reserve(alive_len);
        for (const auto &node : nodes) {
          if (!is_alive(node)) continue;
          const int32_t token = static_cast<int32_t>(node.val);
          for (uint32_t r = 0; r < node.run; ++r) {
            buffer.push_back(token);
          }
          summary.encoded_min_token = std::min(summary.encoded_min_token, token);
          summary.encoded_max_token = std::max(summary.encoded_max_token, token);
        }
        if (!buffer.empty()) {
          out.write(reinterpret_cast<const char *>(buffer.data()),
                    static_cast<std::streamsize>(buffer.size() * sizeof(int32_t)));
        }
        summary.encoded_tokens += alive_len;
        summary.encoded_records += 1;
      }
    }
    if (summary.encoded_records == 0 || summary.encoded_tokens == 0) {
      summary.encoded_min_token = -1;
    }
    out.seekp(0, std::ios::beg);
    if (!out) {
      throw std::runtime_error("failed to seek encoded output header: " + tmp_output);
    }
    write_sample_header(out, summary.encoded_records, summary.encoded_tokens, summary.encoded_max_token);
    out.close();
    if (!out) {
      throw std::runtime_error("failed to finalize encoded output: " + tmp_output);
    }
    atomic_rename(tmp_output, args.encoded_output);
    summary.write_sec += seconds_since(start);
    std::cerr << "encoded compact-int stream written output=" << args.encoded_output
              << " records=" << summary.encoded_records
              << " tokens=" << summary.encoded_tokens
              << " token_range=[" << summary.encoded_min_token << "," << summary.encoded_max_token << "]"
              << " elapsed_sec=" << summary.write_sec << std::endl;
    return;
  }

  for (const auto &shard : trainer.shards) {
    for (const auto &nodes : shard.seqs) {
      for (const auto &node : nodes) {
        if (!is_alive(node)) continue;
        const int32_t token = node.val + args.token_offset;
        for (uint32_t r = 0; r < node.run; ++r) {
          write_token_u16(out, token);
          summary.encoded_tokens += 1;
          summary.encoded_min_token = std::min(summary.encoded_min_token, token);
          summary.encoded_max_token = std::max(summary.encoded_max_token, token);
        }
      }
      if (args.add_eos) {
        write_token_u16(out, args.eos_token_id);
        summary.encoded_tokens += 1;
        summary.encoded_min_token = std::min(summary.encoded_min_token, args.eos_token_id);
        summary.encoded_max_token = std::max(summary.encoded_max_token, args.eos_token_id);
      }
      summary.encoded_records += 1;
    }
  }
  out.close();
  if (!out) {
    throw std::runtime_error("failed to finalize encoded output: " + tmp_output);
  }
  atomic_rename(tmp_output, args.encoded_output);
  summary.write_sec += seconds_since(start);
  if (summary.encoded_records == 0) {
    summary.encoded_min_token = -1;
  }
  std::cerr << "encoded stream written output=" << args.encoded_output
            << " records=" << summary.encoded_records
            << " tokens=" << summary.encoded_tokens
            << " token_range=[" << summary.encoded_min_token << "," << summary.encoded_max_token << "]"
            << " elapsed_sec=" << summary.write_sec << std::endl;
}

Summary run_apply_encode(const Args &args) {
  Summary summary;
  summary.mode = args.encoded_compact_int ? "apply_merges_encode_compact_int" : "apply_merges_encode_u16";
  summary.rank = args.rank;
  summary.world_size = args.world_size;
  summary.threads = args.threads;
  summary.data_shards = args.data_shards > 0 ? args.data_shards : args.threads;
  summary.input_shard = args.input_shard;
  auto load_partition = effective_load_partition(args);
  summary.load_rank = load_partition.first;
  summary.load_world_size = load_partition.second;
  summary.min_frequency = args.min_frequency;
  summary.use_rle = args.use_rle;
  summary.sort_positions = args.sort_positions;
  summary.build_mode = args.build_mode;
  summary.reserve_pairs = args.reserve_pairs;
  summary.global_reserve_pairs = args.global_reserve_pairs > 0 ? args.global_reserve_pairs : args.reserve_pairs;
  summary.token_offset = args.token_offset;
  summary.eos_token_id = args.eos_token_id;
  summary.add_eos = args.add_eos;
  summary.encoded_compact_int = args.encoded_compact_int;
  summary.encoded_output = args.encoded_output;

  auto total_start = Clock::now();
  auto load_start = Clock::now();
  SampleMeta meta;
  auto seqs = load_sample(args.input, summary.load_rank, summary.load_world_size, meta);
  summary.load_sec = seconds_since(load_start);
  summary.input_sequences = seqs.size();
  for (const auto &seq : seqs) summary.input_tokens += seq.size();

  auto split_start = Clock::now();
  auto split = split_for_threads(std::move(seqs), summary.data_shards);
  summary.split_sec = seconds_since(split_start);
  ThreadPoolTrainer trainer;
  BuildTiming build_timing = build_trainer_shards(
      trainer, split, args.threads, args.reserve_pairs, args.use_rle, args.build_mode, meta.max_token + 1);
  summary.build_sec = build_timing.wall_sec;
  summary.build_worker_sum_sec = build_timing.worker_sum_sec;
  summary.build_worker_max_sec = build_timing.worker_max_sec;
  summary.build_node_sum_sec = build_timing.node_sum_sec;
  summary.build_node_max_sec = build_timing.node_max_sec;
  summary.build_index_sum_sec = build_timing.index_sum_sec;
  summary.build_index_max_sec = build_timing.index_max_sec;
  if (args.drop_local_counts_after_init) {
    for (auto &shard : trainer.shards) shard.drop_pair_counts();
  }
  trainer.start_workers(args.threads);

  auto records = read_fixed_apply_records(args);
  summary.num_merges_requested = static_cast<int32_t>(records.size());
  auto train_start = Clock::now();
  for (size_t i = 0; i < records.size(); ++i) {
    const auto &rec = records[i];
    auto apply_start = Clock::now();
    const uint64_t key = pack_pair(rec.left, rec.right);
    auto stats = trainer.apply(key, rec.new_id, rec.step, args.sort_positions);
    summary.apply_sec += seconds_since(apply_start);
    uint64_t real_merges = 0;
    uint64_t positions_seen = 0;
    double sort_sec = 0.0;
    for (const auto &s : stats) {
      real_merges += s.real_merges;
      positions_seen += s.positions_seen;
      sort_sec += s.sort_sec;
    }
    summary.apply_real_merges += real_merges;
    summary.apply_positions_seen += positions_seen;
    summary.sort_sec += sort_sec;
    summary.num_merges_performed += 1;
    if (i == 0 || ((i + 1) % static_cast<size_t>(std::max(1, args.log_every)) == 0)) {
      const double elapsed = seconds_since(train_start);
      std::cerr << "apply-encode progress step=" << (i + 1)
                << " merge_step=" << rec.step
                << " real_merges=" << real_merges
                << " elapsed_sec=" << elapsed;
      write_progress_fields(std::cerr, static_cast<int32_t>(i + 1), static_cast<int32_t>(records.size()), elapsed);
      std::cerr << std::endl;
    }
  }
  for (const auto &shard : trainer.shards) summary.final_tokens += shard.token_count;
  summary.train_sec = seconds_since(train_start);
  trainer.shutdown();
  write_encoded_token_stream(args, trainer, summary);
  summary.total_sec = seconds_since(total_start);
  return summary;
}

/// 单机训练入口：加载样本、构建本地 shard、每轮选择最高频 pair 并增量更新。
Summary run_single(const Args &args) {
  Summary summary;
  summary.mode = "single_incremental_int_bpe";
  summary.rank = 0;
  summary.world_size = 1;
  summary.threads = args.threads;
  summary.data_shards = args.data_shards > 0 ? args.data_shards : args.threads;
  summary.input_shard = false;
  summary.load_rank = 0;
  summary.load_world_size = 1;
  summary.num_merges_requested = args.num_merges;
  summary.min_frequency = args.min_frequency;
  summary.use_rle = args.use_rle;
  summary.sort_positions = args.sort_positions;
  summary.build_mode = args.build_mode;
  summary.reserve_pairs = args.reserve_pairs;
  summary.global_reserve_pairs = args.global_reserve_pairs > 0 ? args.global_reserve_pairs : args.reserve_pairs;
  summary.checkpoint_every = args.checkpoint_every;
  summary.resume_merges = args.resume_merges;
  summary.drop_local_counts_after_init = args.drop_local_counts_after_init;
  summary.stream_local_counts = args.stream_local_counts;
  summary.stream_local_deltas = args.stream_local_deltas;

  auto total_start = Clock::now();
  auto load_start = Clock::now();
  SampleMeta meta;
  auto seqs = load_sample(args.input, 0, 1, meta);
  validate_training_token_capacity(meta.max_token, args.num_merges, args.input);
  summary.load_sec = seconds_since(load_start);
  summary.input_sequences = seqs.size();
  for (const auto &seq : seqs) summary.input_tokens += seq.size();

  auto split_start = Clock::now();
  auto split = split_for_threads(std::move(seqs), summary.data_shards);
  summary.split_sec = seconds_since(split_start);
  ThreadPoolTrainer trainer;
  BuildTiming build_timing = build_trainer_shards(
      trainer, split, args.threads, args.reserve_pairs, args.use_rle, args.build_mode, meta.max_token + 1);
  summary.build_sec = build_timing.wall_sec;
  summary.build_worker_sum_sec = build_timing.worker_sum_sec;
  summary.build_worker_max_sec = build_timing.worker_max_sec;
  summary.build_node_sum_sec = build_timing.node_sum_sec;
  summary.build_node_max_sec = build_timing.node_max_sec;
  summary.build_index_sum_sec = build_timing.index_sum_sec;
  summary.build_index_max_sec = build_timing.index_max_sec;
  trainer.start_workers(args.threads);

  flat_hash_map<uint64_t, int64_t> global_counts;
  global_counts.reserve(summary.global_reserve_pairs);
  std::priority_queue<Candidate, std::vector<Candidate>, CandidateLess> pq;
  auto reduce_start = Clock::now();
  for (auto &shard : trainer.shards) {
    for (const auto &item : shard.pair2cnt) {
      global_counts[item.first] += item.second;
    }
  }
  summary.local_initial_sec = seconds_since(reduce_start);
  summary.initial_unique_pairs = global_counts.size();
  for (const auto &item : global_counts) {
    if (item.second >= args.min_frequency) pq.push({item.first, item.second});
  }
  summary.initial_reduce_sec = seconds_since(reduce_start);

  int32_t next_id = meta.max_token + 1;
  auto train_start = Clock::now();
  for (int32_t step = 0; step < args.num_merges; ++step) {
    auto step_start = Clock::now();
    StepProfile profile;
    profile.step = step;
    auto select_start = Clock::now();
    Candidate best = select_best(global_counts, pq, args.min_frequency);
    profile.select_sec = seconds_since(select_start);
    summary.select_sec += profile.select_sec;
    if (best.count < args.min_frequency) break;
    profile.left = unpack_left(best.key);
    profile.right = unpack_right(best.key);
    profile.new_id = next_id;
    profile.freq = best.count;

    auto apply_start = Clock::now();
    auto stats = trainer.apply(best.key, next_id, step, args.sort_positions);
    profile.apply_wall_sec = seconds_since(apply_start);
    summary.apply_sec += profile.apply_wall_sec;

    auto delta_start = Clock::now();
    uint64_t real_merges = 0;
    uint64_t changed_unique = 0;
    uint64_t positions_seen = 0;
    double sort_sec = 0.0;
    auto local_delta_start = Clock::now();
    for (auto &shard : trainer.shards) {
      for (const auto &item : shard.delta) {
        apply_delta(global_counts, pq, item.first, item.second, args.min_frequency);
      }
      changed_unique += shard.delta.size();
    }
    for (const auto &s : stats) {
      real_merges += s.real_merges;
      positions_seen += s.positions_seen;
      sort_sec += s.sort_sec;
      profile.worker_apply_sum_sec += s.apply_sec;
      profile.worker_apply_max_sec = std::max(profile.worker_apply_max_sec, s.apply_sec);
    }
    profile.local_delta_build_sec = seconds_since(local_delta_start);
    summary.apply_real_merges += real_merges;
    summary.apply_positions_seen += positions_seen;
    summary.sort_sec += sort_sec;
    profile.real_merges = real_merges;
    profile.positions_seen = positions_seen;
    profile.shard_delta_unique_sum = changed_unique;
    profile.local_delta_unique = changed_unique;
    profile.sort_sum_sec = sort_sec;
    profile.global_unique = global_counts.size();
    profile.delta_reduce_sec = seconds_since(delta_start);
    profile.delta_local_apply_sec = profile.delta_reduce_sec;
    profile.step_wall_sec = seconds_since(step_start);
    summary.delta_reduce_sec += profile.delta_reduce_sec;
    summary.merges.push_back({step, unpack_left(best.key), unpack_right(best.key), next_id, best.count, real_merges, positions_seen});
    summary.profiles.push_back(profile);
    summary.num_merges_performed += 1;
    next_id += 1;
    maybe_write_checkpoint(args, summary, summary.num_merges_performed);

    if (step == 0 || (step + 1) % std::max(1, args.log_every) == 0) {
      const double elapsed = seconds_since(total_start);
      std::cerr << "single progress step=" << (step + 1)
                << " freq=" << best.count
                << " real_merges=" << real_merges
                << " changed_unique=" << changed_unique
                << " global_unique=" << global_counts.size()
                << " elapsed_sec=" << elapsed;
      write_progress_fields(std::cerr, step + 1, args.num_merges, elapsed);
      std::cerr << std::endl;
    }
  }
  for (auto &shard : trainer.shards) {
    summary.final_tokens += shard.token_count;
  }
  summary.final_unique_pairs = global_counts.size();
  summary.total_sec = seconds_since(total_start);
  return summary;
}

/// Filesystem-distributed training: exchange counts, merge rules and deltas through shared files.
Summary run_dist(const Args &args) {
  Summary summary;
  summary.mode = "distributed_incremental_int_bpe";
  summary.rank = args.rank;
  summary.world_size = args.world_size;
  summary.threads = args.threads;
  summary.data_shards = args.data_shards > 0 ? args.data_shards : args.threads;
  summary.input_shard = args.input_shard;
  auto load_partition = effective_load_partition(args);
  summary.load_rank = load_partition.first;
  summary.load_world_size = load_partition.second;
  summary.num_merges_requested = args.num_merges;
  summary.min_frequency = args.min_frequency;
  summary.use_rle = args.use_rle;
  summary.sort_positions = args.sort_positions;
  summary.build_mode = args.build_mode;
  summary.reserve_pairs = args.reserve_pairs;
  summary.global_reserve_pairs = args.global_reserve_pairs > 0 ? args.global_reserve_pairs : args.reserve_pairs;
  summary.checkpoint_every = args.checkpoint_every;
  summary.resume_merges = args.resume_merges;
  summary.drop_local_counts_after_init = args.drop_local_counts_after_init;
  summary.stream_local_counts = args.stream_local_counts;
  summary.stream_local_deltas = args.stream_local_deltas;

  ensure_dir(args.out_dir);
  ensure_dir(join_path(args.out_dir, "sync"));
  const std::string sync_dir = join_path(args.out_dir, "sync");
  auto total_start = Clock::now();

  auto load_start = Clock::now();
  SampleMeta meta;
  auto seqs = load_sample(args.input, summary.load_rank, summary.load_world_size, meta);
  validate_training_token_capacity(meta.max_token, args.num_merges, args.input);
  summary.load_sec = seconds_since(load_start);
  summary.input_sequences = seqs.size();
  for (const auto &seq : seqs) summary.input_tokens += seq.size();

  auto split_start = Clock::now();
  auto split = split_for_threads(std::move(seqs), summary.data_shards);
  summary.split_sec = seconds_since(split_start);
  ThreadPoolTrainer trainer;
  BuildTiming build_timing = build_trainer_shards(
      trainer, split, args.threads, args.reserve_pairs, args.use_rle, args.build_mode, meta.max_token + 1);
  summary.build_sec = build_timing.wall_sec;
  summary.build_worker_sum_sec = build_timing.worker_sum_sec;
  summary.build_worker_max_sec = build_timing.worker_max_sec;
  summary.build_node_sum_sec = build_timing.node_sum_sec;
  summary.build_node_max_sec = build_timing.node_max_sec;
  summary.build_index_sum_sec = build_timing.index_sum_sec;
  summary.build_index_max_sec = build_timing.index_max_sec;
  trainer.start_workers(args.threads);

  flat_hash_map<uint64_t, int64_t> local_initial;
  local_initial.reserve(args.reserve_pairs);
  auto local_initial_start = Clock::now();
  for (auto &shard : trainer.shards) {
    for (const auto &item : shard.pair2cnt) {
      local_initial[item.first] += item.second;
    }
  }
  summary.local_initial_sec = seconds_since(local_initial_start);
  summary.initial_unique_pairs = local_initial.size();
  auto write_start = Clock::now();
  write_counts_file(join_path(sync_dir, "init_rank_" + std::to_string(args.rank) + ".bin"), local_initial);
  summary.write_sec += seconds_since(write_start);
  local_initial.clear();

  flat_hash_map<uint64_t, int64_t> global_counts;
  std::priority_queue<Candidate, std::vector<Candidate>, CandidateLess> pq;
  if (args.rank == 0) {
    global_counts.reserve(summary.global_reserve_pairs);
    auto reduce_start = Clock::now();
    for (int r = 0; r < args.world_size; ++r) {
      const std::string p = join_path(sync_dir, "init_rank_" + std::to_string(r) + ".bin");
      wait_for_file(p, args.sync_timeout_sec, summary.sync_wait_sec);
      auto counts = read_counts_file(p);
      for (const auto &item : counts) {
        global_counts[item.first] += item.second;
      }
    }
    for (const auto &item : global_counts) {
      if (item.second >= args.min_frequency) pq.push({item.first, item.second});
    }
    summary.initial_reduce_sec = seconds_since(reduce_start);
  } else {
    for (int r = 0; r < args.world_size; ++r) {
      const std::string p = join_path(sync_dir, "init_rank_" + std::to_string(r) + ".bin");
      wait_for_file(p, args.sync_timeout_sec, summary.sync_wait_sec);
    }
  }

  int32_t next_id = meta.max_token + 1;
  auto train_start = Clock::now();
  flat_hash_map<uint64_t, int64_t> local_delta;
  local_delta.reserve(kDeltaReserve);
  for (int32_t step = 0; step < args.num_merges; ++step) {
    auto step_start = Clock::now();
    StepProfile profile;
    profile.step = step;
    std::string merge_path = join_path(sync_dir, "merge_step_" + std::to_string(step) + ".bin");
    uint64_t best_key = 0;
    int64_t best_freq = -1;
    bool stop = false;
    if (args.rank == 0) {
      auto select_start = Clock::now();
      Candidate best = select_best(global_counts, pq, args.min_frequency);
      profile.select_sec = seconds_since(select_start);
      summary.select_sec += profile.select_sec;
      best_key = best.key;
      best_freq = best.count;
      stop = best.count < args.min_frequency;
      write_start = Clock::now();
      write_merge_file(merge_path, stop, step, best_key, next_id, best_freq);
      profile.merge_sync_sec = seconds_since(write_start);
      summary.write_sec += profile.merge_sync_sec;
    } else {
      auto merge_wait_start = Clock::now();
      wait_for_file(merge_path, args.sync_timeout_sec, summary.sync_wait_sec);
      int32_t read_step = 0;
      int32_t read_new_id = 0;
      read_merge_file(merge_path, stop, read_step, best_key, read_new_id, best_freq);
      profile.merge_sync_sec = seconds_since(merge_wait_start);
      next_id = read_new_id;
    }
    if (stop) break;
    profile.left = unpack_left(best_key);
    profile.right = unpack_right(best_key);
    profile.new_id = next_id;
    profile.freq = best_freq;

    auto apply_start = Clock::now();
    auto stats = trainer.apply(best_key, next_id, step, args.sort_positions);
    profile.apply_wall_sec = seconds_since(apply_start);
    summary.apply_sec += profile.apply_wall_sec;

    auto local_delta_start = Clock::now();
    local_delta.clear();
    uint64_t real_merges = 0;
    uint64_t changed_unique = 0;
    uint64_t positions_seen = 0;
    double sort_sec = 0.0;
    for (auto &shard : trainer.shards) {
      for (const auto &item : shard.delta) {
        local_delta[item.first] += item.second;
      }
      changed_unique += shard.delta.size();
    }
    for (const auto &s : stats) {
      real_merges += s.real_merges;
      positions_seen += s.positions_seen;
      sort_sec += s.sort_sec;
      profile.worker_apply_sum_sec += s.apply_sec;
      profile.worker_apply_max_sec = std::max(profile.worker_apply_max_sec, s.apply_sec);
    }
    profile.local_delta_build_sec = seconds_since(local_delta_start);
    summary.apply_real_merges += real_merges;
    summary.apply_positions_seen += positions_seen;
    summary.sort_sec += sort_sec;

    write_start = Clock::now();
    const std::string delta_path = join_path(
        sync_dir,
        "delta_step_" + std::to_string(step) + "_rank_" + std::to_string(args.rank) + ".bin");
    write_counts_file(delta_path, local_delta);
    profile.delta_send_sec = seconds_since(write_start);
    summary.write_sec += profile.delta_send_sec;
    summary.merges.push_back({step, unpack_left(best_key), unpack_right(best_key), next_id, best_freq, real_merges, positions_seen});

    if (args.rank == 0) {
      auto reduce_start = Clock::now();
      for (int r = 0; r < args.world_size; ++r) {
        const std::string p = join_path(sync_dir, "delta_step_" + std::to_string(step) + "_rank_" + std::to_string(r) + ".bin");
        wait_for_file(p, args.sync_timeout_sec, summary.sync_wait_sec);
        auto delta = read_counts_file(p);
        auto apply_start = Clock::now();
        for (const auto &item : delta) {
          apply_delta(global_counts, pq, item.first, item.second, args.min_frequency);
        }
        profile.delta_recv_apply_sec += seconds_since(apply_start);
        remove_file(p);
      }
      profile.delta_reduce_sec = seconds_since(reduce_start);
      summary.delta_reduce_sec += profile.delta_reduce_sec;
      if (step == 0 || (step + 1) % std::max(1, args.log_every) == 0) {
        const double train_elapsed = seconds_since(train_start);
        const double elapsed = seconds_since(total_start);
        std::cerr << "dist progress step=" << (step + 1)
                  << " freq=" << best_freq
                  << " rank0_real_merges=" << real_merges
                  << " rank0_changed_unique=" << changed_unique
                  << " global_unique=" << global_counts.size()
                  << " train_elapsed_sec=" << train_elapsed
                  << " elapsed_sec=" << elapsed;
        write_progress_fields(std::cerr, step + 1, args.num_merges, train_elapsed);
        std::cerr << std::endl;
      }
      if (step > 0) remove_file(join_path(sync_dir, "merge_step_" + std::to_string(step - 1) + ".bin"));
    }
    profile.real_merges = real_merges;
    profile.positions_seen = positions_seen;
    profile.shard_delta_unique_sum = changed_unique;
    profile.local_delta_unique = local_delta.size();
    profile.sort_sum_sec = sort_sec;
    profile.global_unique = args.rank == 0 ? global_counts.size() : 0;
    profile.step_wall_sec = seconds_since(step_start);
    summary.profiles.push_back(profile);
    summary.num_merges_performed += 1;
    next_id += 1;
    maybe_write_checkpoint(args, summary, summary.num_merges_performed);
  }

  for (auto &shard : trainer.shards) summary.final_tokens += shard.token_count;
  summary.final_unique_pairs = args.rank == 0 ? global_counts.size() : 0;
  summary.train_sec = seconds_since(train_start);
  summary.total_sec = seconds_since(total_start);
  return summary;
}

/// TCP 分布式训练入口：rank0 常驻汇总全局计数，并通过 socket 广播 merge、接收 delta。
Summary run_dist_tcp(const Args &args) {
  Summary summary;
  summary.mode = "distributed_tcp_incremental_int_bpe";
  summary.rank = args.rank;
  summary.world_size = args.world_size;
  summary.threads = args.threads;
  summary.data_shards = args.data_shards > 0 ? args.data_shards : args.threads;
  summary.input_shard = args.input_shard;
  auto load_partition = effective_load_partition(args);
  summary.load_rank = load_partition.first;
  summary.load_world_size = load_partition.second;
  summary.num_merges_requested = args.num_merges;
  summary.min_frequency = args.min_frequency;
  summary.use_rle = args.use_rle;
  summary.sort_positions = args.sort_positions;
  summary.build_mode = args.build_mode;
  summary.reserve_pairs = args.reserve_pairs;
  summary.global_reserve_pairs = args.global_reserve_pairs > 0 ? args.global_reserve_pairs : args.reserve_pairs;
  summary.checkpoint_every = args.checkpoint_every;
  summary.resume_merges = args.resume_merges;
  summary.drop_local_counts_after_init = args.drop_local_counts_after_init;
  summary.stream_local_counts = args.stream_local_counts;
  summary.stream_local_deltas = args.stream_local_deltas;

  if (args.tcp_rendezvous.empty()) {
    throw std::runtime_error("--tcp-rendezvous is required for --mode dist-tcp");
  }
  ensure_dir(args.out_dir);
  auto total_start = Clock::now();
  TcpStats net_stats;

  int listen_fd = -1;
  std::vector<int> peer_fds(args.world_size, -1);
  int server_fd = -1;
  auto connect_start = Clock::now();
  if (args.rank == 0) {
    int actual_port = 0;
    listen_fd = create_listener(args.tcp_port, actual_port);
    const std::string host = get_preferred_ip();
    write_rendezvous(args.tcp_rendezvous, host, actual_port);
    std::cerr << "tcp rank0 listening host=" << host << " port=" << actual_port << std::endl;
    for (int i = 1; i < args.world_size; ++i) {
      sockaddr_in addr{};
      socklen_t addr_len = sizeof(addr);
      int fd = ::accept(listen_fd, reinterpret_cast<sockaddr *>(&addr), &addr_len);
      if (fd < 0) throw std::runtime_error(errno_message("accept failed"));
      set_tcp_no_delay(fd);
      int32_t peer_rank = -1;
      recv_value(fd, peer_rank, net_stats);
      if (peer_rank <= 0 || peer_rank >= args.world_size) {
        ::close(fd);
        throw std::runtime_error("invalid peer rank on tcp connect: " + std::to_string(peer_rank));
      }
      if (peer_fds[peer_rank] >= 0) {
        ::close(fd);
        throw std::runtime_error("duplicate peer rank on tcp connect: " + std::to_string(peer_rank));
      }
      peer_fds[peer_rank] = fd;
      std::cerr << "tcp rank0 accepted rank=" << peer_rank << std::endl;
    }
  } else {
    wait_for_file(args.tcp_rendezvous, args.sync_timeout_sec, summary.sync_wait_sec);
    auto endpoint = read_rendezvous(args.tcp_rendezvous);
    server_fd = connect_with_retry(endpoint.first, endpoint.second, args.sync_timeout_sec, summary.sync_wait_sec);
    int32_t rank_i32 = args.rank;
    send_value(server_fd, rank_i32, net_stats);
    std::cerr << "tcp rank=" << args.rank << " connected host=" << endpoint.first << " port=" << endpoint.second << std::endl;
  }
  summary.connect_sec = seconds_since(connect_start);

  auto load_start = Clock::now();
  SampleMeta meta;
  auto seqs = load_sample(args.input, summary.load_rank, summary.load_world_size, meta);
  validate_training_token_capacity(meta.max_token, args.num_merges, args.input);
  summary.load_sec = seconds_since(load_start);
  summary.input_sequences = seqs.size();
  for (const auto &seq : seqs) summary.input_tokens += seq.size();

  auto split_start = Clock::now();
  auto split = split_for_threads(std::move(seqs), summary.data_shards);
  summary.split_sec = seconds_since(split_start);
  ThreadPoolTrainer trainer;
  BuildTiming build_timing = build_trainer_shards(
      trainer, split, args.threads, args.reserve_pairs, args.use_rle, args.build_mode, meta.max_token + 1);
  summary.build_sec = build_timing.wall_sec;
  summary.build_worker_sum_sec = build_timing.worker_sum_sec;
  summary.build_worker_max_sec = build_timing.worker_max_sec;
  summary.build_node_sum_sec = build_timing.node_sum_sec;
  summary.build_node_max_sec = build_timing.node_max_sec;
  summary.build_index_sum_sec = build_timing.index_sum_sec;
  summary.build_index_max_sec = build_timing.index_max_sec;
  trainer.start_workers(args.threads);

  auto local_initial_start = Clock::now();
  flat_hash_map<uint64_t, int64_t> local_initial;
  if (!args.stream_local_counts) {
    local_initial.reserve(args.reserve_pairs);
    for (auto &shard : trainer.shards) {
      for (const auto &item : shard.pair2cnt) {
        local_initial[item.first] += item.second;
      }
    }
  }
  summary.local_initial_sec = seconds_since(local_initial_start);
  summary.initial_unique_pairs = args.stream_local_counts ? 0 : local_initial.size();

  flat_hash_map<uint64_t, int64_t> global_counts;
  std::priority_queue<Candidate, std::vector<Candidate>, CandidateLess> pq;
  if (args.rank == 0) {
    global_counts.reserve(summary.global_reserve_pairs);
    auto reduce_start = Clock::now();
    if (args.stream_local_counts) {
      auto local_apply_start = Clock::now();
      for (const auto &shard : trainer.shards) {
        shard.for_each_pair_count([&](uint64_t key, int64_t value) {
          auto inserted = global_counts.emplace(key, value);
          if (!inserted.second) inserted.first->second += value;
        });
      }
      summary.local_initial_sec = seconds_since(local_apply_start);
      summary.initial_unique_pairs = global_counts.size();
    } else {
      for (const auto &item : local_initial) {
        auto inserted = global_counts.emplace(item.first, item.second);
        if (!inserted.second) inserted.first->second += item.second;
      }
    }
    for (int r = 1; r < args.world_size; ++r) {
      recv_counts_socket_for_each(peer_fds[r], net_stats, [&](uint64_t key, int64_t value) {
        auto inserted = global_counts.emplace(key, value);
        if (!inserted.second) inserted.first->second += value;
      });
    }
    for (const auto &item : global_counts) {
      if (item.second >= args.min_frequency) pq.push({item.first, item.second});
    }
    summary.initial_reduce_sec = seconds_since(reduce_start);
  } else {
    auto write_start = Clock::now();
    if (args.stream_local_counts) {
      send_initial_counts_socket_from_shards(server_fd, trainer.shards, net_stats);
    } else {
      send_counts_socket(server_fd, local_initial, net_stats);
    }
    summary.write_sec += seconds_since(write_start);
  }
  local_initial.clear();
  flat_hash_map<uint64_t, int64_t>().swap(local_initial);
  if (args.drop_local_counts_after_init) {
    for (auto &shard : trainer.shards) shard.drop_pair_counts();
  }

  auto barrier_start = Clock::now();
  if (args.rank == 0) {
    tcp_barrier_rank0(peer_fds, args.world_size, net_stats);
  } else {
    tcp_barrier_worker(server_fd, net_stats);
  }
  summary.barrier_sec = seconds_since(barrier_start);
  auto train_start = Clock::now();

  int32_t next_id = meta.max_token + 1;
  flat_hash_map<uint64_t, int64_t> local_delta;
  local_delta.reserve(kDeltaReserve);
  auto resume_merges = load_resume_merges_for_rank(args, meta.max_token + 1);
  summary.resume_steps = static_cast<int32_t>(resume_merges.size());
  if (summary.resume_steps > args.num_merges) {
    throw std::runtime_error("--resume-merges contains more rows than --num-merges");
  }
  if (summary.resume_steps > 0) {
    std::cerr << "dist-tcp resume rank=" << args.rank
              << " replay_steps=" << summary.resume_steps
              << " from=" << args.resume_merges << std::endl;
  }
  for (int32_t step = 0; step < args.num_merges; ++step) {
    auto step_start = Clock::now();
    StepProfile profile;
    profile.step = step;
    uint64_t best_key = 0;
    int64_t best_freq = -1;
    bool stop = false;
    if (args.rank == 0) {
      auto select_start = Clock::now();
      Candidate best;
      if (step < summary.resume_steps) {
        best.key = pack_pair(resume_merges[step].left, resume_merges[step].right);
        best.count = resume_merges[step].freq;
        Candidate current_best = select_best(global_counts, pq, args.min_frequency);
        if (current_best.key != best.key || current_best.count != best.count) {
          std::cerr << "WARNING: resume step " << step
                    << " current_best=(" << unpack_left(current_best.key) << "," << unpack_right(current_best.key)
                    << ") freq=" << current_best.count
                    << " checkpoint=(" << resume_merges[step].left << "," << resume_merges[step].right
                    << ") freq=" << best.count << std::endl;
          throw std::runtime_error("resume checkpoint does not match current input/parameters at step " + std::to_string(step));
        }
      } else {
        best = select_best(global_counts, pq, args.min_frequency);
      }
      profile.select_sec = seconds_since(select_start);
      summary.select_sec += profile.select_sec;
      best_key = best.key;
      best_freq = best.count;
      stop = step >= summary.resume_steps && best.count < args.min_frequency;
      MergeWire msg;
      msg.stop = stop ? 1 : 0;
      msg.step = step;
      msg.key = best_key;
      msg.new_id = step < summary.resume_steps ? resume_merges[step].new_id : next_id;
      msg.freq = best_freq;
      auto write_start = Clock::now();
      for (int r = 1; r < args.world_size; ++r) {
        send_merge_socket(peer_fds[r], msg, net_stats);
      }
      profile.merge_sync_sec = seconds_since(write_start);
      summary.write_sec += profile.merge_sync_sec;
      next_id = msg.new_id;
    } else {
      auto wait_start = Clock::now();
      MergeWire msg = recv_merge_socket(server_fd, net_stats);
      profile.merge_sync_sec = seconds_since(wait_start);
      summary.sync_wait_sec += profile.merge_sync_sec;
      stop = msg.stop != 0;
      if (msg.step != step) {
        throw std::runtime_error("tcp merge step mismatch rank=" + std::to_string(args.rank));
      }
      best_key = msg.key;
      best_freq = msg.freq;
      next_id = msg.new_id;
    }
    if (stop) break;
    if (step < summary.resume_steps) {
      const MergeRecord &expected = resume_merges[step];
      if (expected.new_id != next_id ||
          expected.left != unpack_left(best_key) ||
          expected.right != unpack_right(best_key)) {
        throw std::runtime_error("resume merge mismatch at step " + std::to_string(step));
      }
    }
    profile.left = unpack_left(best_key);
    profile.right = unpack_right(best_key);
    profile.new_id = next_id;
    profile.freq = best_freq;

    auto apply_start = Clock::now();
    auto stats = trainer.apply(best_key, next_id, step, args.sort_positions);
    profile.apply_wall_sec = seconds_since(apply_start);
    summary.apply_sec += profile.apply_wall_sec;

    auto local_delta_start = Clock::now();
    local_delta.clear();
    uint64_t real_merges = 0;
    uint64_t changed_unique = 0;
    uint64_t positions_seen = 0;
    double sort_sec = 0.0;
    if (!args.stream_local_deltas || args.rank == 0) {
      for (auto &shard : trainer.shards) {
        for (const auto &item : shard.delta) {
          local_delta[item.first] += item.second;
        }
        changed_unique += shard.delta.size();
      }
    } else {
      for (auto &shard : trainer.shards) {
        changed_unique += shard.delta.size();
      }
    }
    for (const auto &s : stats) {
      real_merges += s.real_merges;
      positions_seen += s.positions_seen;
      sort_sec += s.sort_sec;
      profile.worker_apply_sum_sec += s.apply_sec;
      profile.worker_apply_max_sec = std::max(profile.worker_apply_max_sec, s.apply_sec);
    }
    profile.local_delta_build_sec = seconds_since(local_delta_start);
    summary.apply_real_merges += real_merges;
    summary.apply_positions_seen += positions_seen;
    summary.sort_sec += sort_sec;
    summary.merges.push_back({step, unpack_left(best_key), unpack_right(best_key), next_id, best_freq, real_merges, positions_seen});

    if (args.rank == 0) {
      auto reduce_start = Clock::now();
      auto local_apply_start = Clock::now();
      if (!args.stream_local_deltas) {
        for (const auto &item : local_delta) {
          apply_delta(global_counts, pq, item.first, item.second, args.min_frequency);
        }
      } else {
        for (const auto &shard : trainer.shards) {
          shard.for_each_delta([&](uint64_t key, int64_t value) {
            apply_delta(global_counts, pq, key, value, args.min_frequency);
          });
        }
      }
      profile.delta_local_apply_sec = seconds_since(local_apply_start);
      for (int r = 1; r < args.world_size; ++r) {
        auto remote_apply_start = Clock::now();
        recv_counts_socket_for_each(peer_fds[r], net_stats, [&](uint64_t key, int64_t value) {
          apply_delta(global_counts, pq, key, value, args.min_frequency);
        });
        profile.delta_recv_apply_sec += seconds_since(remote_apply_start);
      }
      profile.delta_reduce_sec = seconds_since(reduce_start);
      summary.delta_reduce_sec += profile.delta_reduce_sec;
      if (step == 0 || (step + 1) % std::max(1, args.log_every) == 0) {
        const double train_elapsed = seconds_since(train_start);
        const double elapsed = seconds_since(total_start);
        std::cerr << "dist-tcp progress step=" << (step + 1)
                  << " freq=" << best_freq
                  << " rank0_real_merges=" << real_merges
                  << " rank0_changed_unique=" << changed_unique
                  << " global_unique=" << global_counts.size()
                  << " train_elapsed_sec=" << train_elapsed
                  << " elapsed_sec=" << elapsed;
        write_progress_fields(std::cerr, step + 1, args.num_merges, train_elapsed);
        std::cerr << std::endl;
      }
    } else {
      auto write_start = Clock::now();
      if (args.stream_local_deltas) {
        send_delta_counts_socket_from_shards(server_fd, trainer.shards, net_stats);
      } else {
        send_counts_socket(server_fd, local_delta, net_stats);
      }
      profile.delta_send_sec = seconds_since(write_start);
      summary.write_sec += profile.delta_send_sec;
    }
    profile.real_merges = real_merges;
    profile.positions_seen = positions_seen;
    profile.shard_delta_unique_sum = changed_unique;
    profile.local_delta_unique = args.stream_local_deltas && args.rank != 0 ? changed_unique : local_delta.size();
    profile.sort_sum_sec = sort_sec;
    profile.global_unique = args.rank == 0 ? global_counts.size() : 0;
    profile.step_wall_sec = seconds_since(step_start);
    summary.profiles.push_back(profile);
    summary.num_merges_performed += 1;
    next_id += 1;
    maybe_write_checkpoint(args, summary, summary.num_merges_performed);
  }

  for (auto &shard : trainer.shards) summary.final_tokens += shard.token_count;
  summary.final_unique_pairs = args.rank == 0 ? global_counts.size() : 0;
  summary.train_sec = seconds_since(train_start);
  summary.total_sec = seconds_since(total_start);
  summary.network_bytes_sent = net_stats.sent;
  summary.network_bytes_recv = net_stats.received;
  if (listen_fd >= 0) ::close(listen_fd);
  if (server_fd >= 0) ::close(server_fd);
  for (int fd : peer_fds) {
    if (fd >= 0) ::close(fd);
  }
  return summary;
}

/// 解析命令行参数，并检查模式、rank、线程数和构建模式是否合法。
Args parse_args(int argc, char **argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    std::string key = argv[i];
    auto need_value = [&](const std::string &name) -> std::string {
      if (i + 1 >= argc) throw std::runtime_error("missing value for " + name);
      return argv[++i];
    };
    if (key == "--mode") args.mode = need_value(key);
    else if (key == "--input") args.input = need_value(key);
    else if (key == "--out-dir") args.out_dir = need_value(key);
    else if (key == "--num-merges") args.num_merges = std::stoi(need_value(key));
    else if (key == "--min-frequency") args.min_frequency = std::stoll(need_value(key));
    else if (key == "--threads") args.threads = std::stoi(need_value(key));
    else if (key == "--shards") args.data_shards = std::stoi(need_value(key));
    else if (key == "--data-shards") args.data_shards = std::stoi(need_value(key));
    else if (key == "--rank") args.rank = std::stoi(need_value(key));
    else if (key == "--world-size") args.world_size = std::stoi(need_value(key));
    else if (key == "--load-rank") args.load_rank = std::stoi(need_value(key));
    else if (key == "--load-world-size") args.load_world_size = std::stoi(need_value(key));
    else if (key == "--log-every") args.log_every = std::stoi(need_value(key));
    else if (key == "--sync-timeout-sec") args.sync_timeout_sec = std::stoi(need_value(key));
    else if (key == "--reserve-pairs") args.reserve_pairs = static_cast<uint64_t>(std::stoull(need_value(key)));
    else if (key == "--global-reserve-pairs") args.global_reserve_pairs = static_cast<uint64_t>(std::stoull(need_value(key)));
    else if (key == "--checkpoint-every") args.checkpoint_every = std::stoi(need_value(key));
    else if (key == "--resume-merges") args.resume_merges = need_value(key);
    else if (key == "--apply-merges") args.apply_merges = need_value(key);
    else if (key == "--apply-skip") args.apply_skip = std::stoi(need_value(key));
    else if (key == "--domain-runs") args.domain_runs = need_value(key);
    else if (key == "--encoded-output") args.encoded_output = need_value(key);
    else if (key == "--token-offset") args.token_offset = std::stoi(need_value(key));
    else if (key == "--eos-token-id") args.eos_token_id = std::stoi(need_value(key));
    else if (key == "--tcp-rendezvous") args.tcp_rendezvous = need_value(key);
    else if (key == "--tcp-port") args.tcp_port = std::stoi(need_value(key));
    else if (key == "--input-shard") args.input_shard = true;
    else if (key == "--no-input-shard") args.input_shard = false;
    else if (key == "--sort-positions") args.sort_positions = true;
    else if (key == "--no-sort-positions") args.sort_positions = false;
    else if (key == "--rle") args.use_rle = true;
    else if (key == "--no-rle") args.use_rle = false;
    else if (key == "--add-eos") args.add_eos = true;
    else if (key == "--no-eos") args.add_eos = false;
    else if (key == "--no-add-eos") args.add_eos = false;
    else if (key == "--encoded-compact-int") args.encoded_compact_int = true;
    else if (key == "--no-encoded-compact-int") args.encoded_compact_int = false;
    else if (key == "--overwrite-output") args.overwrite_output = true;
    else if (key == "--drop-local-counts-after-init") args.drop_local_counts_after_init = true;
    else if (key == "--keep-local-counts-after-init") args.drop_local_counts_after_init = false;
    else if (key == "--stream-local-counts") args.stream_local_counts = true;
    else if (key == "--no-stream-local-counts") args.stream_local_counts = false;
    else if (key == "--stream-local-deltas") args.stream_local_deltas = true;
    else if (key == "--no-stream-local-deltas") args.stream_local_deltas = false;
    else if (key == "--build-mode") args.build_mode = need_value(key);
    else throw std::runtime_error("unknown argument: " + key);
  }
  if (args.input.empty()) throw std::runtime_error("--input is required");
  if (args.out_dir.empty()) throw std::runtime_error("--out-dir is required");
  if (args.mode != "single" && args.mode != "dist" && args.mode != "dist-tcp" &&
      args.mode != "apply-stats" && args.mode != "apply-encode") {
    throw std::runtime_error("unsupported --mode: " + args.mode);
  }
  if (args.num_merges < 0) throw std::runtime_error("--num-merges must be non-negative");
  if (args.min_frequency < 1) throw std::runtime_error("--min-frequency must be positive");
  if (args.threads < 1) throw std::runtime_error("--threads must be positive");
  if (args.data_shards < 0) throw std::runtime_error("--shards must be non-negative");
  if (args.world_size < 1) throw std::runtime_error("--world-size must be positive");
  if (args.rank < 0 || args.rank >= args.world_size) throw std::runtime_error("invalid --rank/--world-size");
  if (args.log_every < 1) throw std::runtime_error("--log-every must be positive");
  if (args.sync_timeout_sec < 1) throw std::runtime_error("--sync-timeout-sec must be positive");
  if (args.checkpoint_every < 0) throw std::runtime_error("--checkpoint-every must be non-negative");
  if (args.apply_skip < 0) throw std::runtime_error("--apply-skip must be non-negative");
  if (args.token_offset < 0) throw std::runtime_error("--token-offset must be non-negative");
  if (args.eos_token_id < 0 || args.eos_token_id > std::numeric_limits<uint16_t>::max()) {
    throw std::runtime_error("--eos-token-id must fit uint16");
  }
  if (!args.resume_merges.empty() && args.mode != "dist-tcp") {
    throw std::runtime_error("--resume-merges is currently supported only with --mode dist-tcp");
  }
  if (!args.resume_merges.empty() && !file_exists(args.resume_merges)) {
    throw std::runtime_error("missing resume merges: " + args.resume_merges);
  }
  if (args.mode == "apply-stats") {
    if (args.apply_merges.empty()) throw std::runtime_error("--apply-merges is required with --mode apply-stats");
    if (args.domain_runs.empty()) throw std::runtime_error("--domain-runs is required with --mode apply-stats");
    if (!file_exists(args.apply_merges)) throw std::runtime_error("missing apply merges: " + args.apply_merges);
    if (!file_exists(args.domain_runs)) throw std::runtime_error("missing domain runs: " + args.domain_runs);
    if (args.use_rle) throw std::runtime_error("--mode apply-stats currently requires --no-rle");
  }
  if (args.mode == "apply-encode") {
    if (args.apply_merges.empty()) throw std::runtime_error("--apply-merges is required with --mode apply-encode");
    if (args.encoded_output.empty()) throw std::runtime_error("--encoded-output is required with --mode apply-encode");
    if (!file_exists(args.apply_merges)) throw std::runtime_error("missing apply merges: " + args.apply_merges);
    if (args.use_rle) throw std::runtime_error("--mode apply-encode currently requires --no-rle");
    if (args.encoded_compact_int && args.add_eos) {
      throw std::runtime_error("--encoded-compact-int requires --no-eos");
    }
    if (args.encoded_compact_int && args.token_offset != 0) {
      throw std::runtime_error("--encoded-compact-int requires --token-offset 0");
    }
  } else if (args.encoded_compact_int) {
    throw std::runtime_error("--encoded-compact-int is only valid with --mode apply-encode");
  }
  if (args.tcp_port < 0 || args.tcp_port > 65535) {
    throw std::runtime_error("--tcp-port must be in [0, 65535]");
  }
  if (args.reserve_pairs > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
    throw std::runtime_error("--reserve-pairs exceeds size_t limit");
  }
  if (args.global_reserve_pairs > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
    throw std::runtime_error("--global-reserve-pairs exceeds size_t limit");
  }
  if ((args.load_rank >= 0) != (args.load_world_size > 0)) {
    throw std::runtime_error("--load-rank and --load-world-size must be set together");
  }
  if (args.load_world_size > 0 && (args.load_rank < 0 || args.load_rank >= args.load_world_size)) {
    throw std::runtime_error("invalid --load-rank/--load-world-size");
  }
  if (args.build_mode != "legacy-noreserve" &&
      args.build_mode != "legacy-reserve" &&
      args.build_mode != "legacy-growth-reserve" &&
      args.build_mode != "onepass" &&
      args.build_mode != "onepass-noreserve" &&
      args.build_mode != "onepass-growth-reserve" &&
      args.build_mode != "base-bucket") {
    throw std::runtime_error("invalid --build-mode: " + args.build_mode);
  }
  if (args.use_rle && args.build_mode == "base-bucket") {
    throw std::runtime_error("--build-mode base-bucket is only valid with --no-rle");
  }
  return args;
}

}  // namespace

/// 程序入口：按 mode 调用对应训练流程，并写出 summary、merges 和 profile 文件。
int main(int argc, char **argv) {
  try {
    Args args = parse_args(argc, argv);
    ensure_dir(args.out_dir);
    Summary summary;
    if (args.mode == "single") {
      summary = run_single(args);
      write_summary(join_path(args.out_dir, "summary.json"), summary);
      write_merges(join_path(args.out_dir, "merges.csv"), summary);
      write_profile(join_path(args.out_dir, "profile.csv"), summary);
    } else if (args.mode == "dist") {
      summary = run_dist(args);
      write_summary(join_path(args.out_dir, "summary_rank_" + std::to_string(args.rank) + ".json"), summary);
      write_merges(join_path(args.out_dir, "merges_rank_" + std::to_string(args.rank) + ".csv"), summary);
      write_profile(join_path(args.out_dir, "profile_rank_" + std::to_string(args.rank) + ".csv"), summary);
      if (args.rank == 0) {
        write_summary(join_path(args.out_dir, "summary.json"), summary);
        write_merges(join_path(args.out_dir, "merges.csv"), summary);
        write_profile(join_path(args.out_dir, "profile.csv"), summary);
      }
    } else if (args.mode == "dist-tcp") {
      summary = run_dist_tcp(args);
      write_summary(join_path(args.out_dir, "summary_rank_" + std::to_string(args.rank) + ".json"), summary);
      write_merges(join_path(args.out_dir, "merges_rank_" + std::to_string(args.rank) + ".csv"), summary);
      write_profile(join_path(args.out_dir, "profile_rank_" + std::to_string(args.rank) + ".csv"), summary);
      if (args.rank == 0) {
        write_summary(join_path(args.out_dir, "summary.json"), summary);
        write_merges(join_path(args.out_dir, "merges.csv"), summary);
        write_profile(join_path(args.out_dir, "profile.csv"), summary);
      }
    } else if (args.mode == "apply-stats") {
      summary = run_apply_stats(args);
      write_summary(join_path(args.out_dir, "summary_rank_" + std::to_string(args.rank) + ".json"), summary);
    } else if (args.mode == "apply-encode") {
      summary = run_apply_encode(args);
      write_summary(join_path(args.out_dir, "summary_rank_" + std::to_string(args.rank) + ".json"), summary);
    } else {
      throw std::runtime_error("unsupported --mode: " + args.mode);
    }
    std::cerr << "summary written mode=" << summary.mode
              << " rank=" << summary.rank
              << " merges=" << summary.num_merges_performed
              << " total_sec=" << summary.total_sec << std::endl;
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "ERROR: " << e.what() << std::endl;
    return 1;
  }
}
