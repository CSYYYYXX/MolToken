#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <memory>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

constexpr char SAMPLE_MAGIC[8] = {'Y', 'T', 'T', 'M', 'I', 'B', 'P', 'E'};
constexpr uint32_t SAMPLE_VERSION = 1;

struct Args {
  std::string input;
  std::string output;
  std::string merges_csv;
  std::string summary;
  int32_t skip = 0;
  int32_t limit = 0;
  uint64_t log_every = 100000;
  bool overwrite = false;
};

struct SampleMeta {
  uint64_t nseq = 0;
  uint64_t ntokens = 0;
  int32_t max_token = -1;
};

struct MergeRule {
  int32_t step = 0;
  int32_t left = 0;
  int32_t right = 0;
  int32_t new_id = 0;
};

struct Summary {
  uint64_t input_records = 0;
  uint64_t input_tokens = 0;
  uint64_t output_records = 0;
  uint64_t output_tokens = 0;
  uint64_t real_merges = 0;
  int32_t input_max_token = -1;
  int32_t output_max_token = -1;
  double elapsed_sec = 0.0;
};

double seconds_since(const Clock::time_point &start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}

template <typename T>
void write_raw(std::ofstream &out, const T &value) {
  out.write(reinterpret_cast<const char *>(&value), sizeof(T));
  if (!out) throw std::runtime_error("failed to write binary value");
}

template <typename T>
void read_raw(std::ifstream &in, T &value) {
  in.read(reinterpret_cast<char *>(&value), sizeof(T));
  if (!in) throw std::runtime_error("unexpected EOF");
}

void write_header(std::ofstream &out, const SampleMeta &meta) {
  out.seekp(0);
  out.write(SAMPLE_MAGIC, 8);
  write_raw(out, SAMPLE_VERSION);
  write_raw(out, meta.nseq);
  write_raw(out, meta.ntokens);
  write_raw(out, meta.max_token);
}

SampleMeta read_header(std::ifstream &in) {
  char magic[8];
  in.read(magic, 8);
  if (!in || !std::equal(magic, magic + 8, SAMPLE_MAGIC)) {
    throw std::runtime_error("invalid int shard magic");
  }
  uint32_t version = 0;
  read_raw(in, version);
  if (version != SAMPLE_VERSION) {
    throw std::runtime_error("unsupported int shard version: " + std::to_string(version));
  }
  SampleMeta meta;
  read_raw(in, meta.nseq);
  read_raw(in, meta.ntokens);
  read_raw(in, meta.max_token);
  return meta;
}

bool exists(const std::string &path) {
  std::ifstream in(path, std::ios::binary);
  return in.good();
}

std::vector<std::string> split_csv_line(const std::string &line) {
  std::vector<std::string> fields;
  std::string cur;
  std::stringstream ss(line);
  while (std::getline(ss, cur, ',')) fields.push_back(cur);
  return fields;
}

std::vector<MergeRule> read_merge_rules(const std::string &path, int32_t skip, int32_t limit) {
  if (skip < 0) throw std::runtime_error("--skip must be non-negative");
  if (limit < 0) throw std::runtime_error("--limit must be non-negative");
  std::ifstream in(path);
  if (!in) throw std::runtime_error("failed to open merges csv: " + path);
  std::string line;
  if (!std::getline(in, line)) throw std::runtime_error("empty merges csv: " + path);
  auto header = split_csv_line(line);
  int step_col = -1;
  int left_col = -1;
  int right_col = -1;
  int new_col = -1;
  for (int i = 0; i < static_cast<int>(header.size()); ++i) {
    if (header[i] == "step") step_col = i;
    else if (header[i] == "left") left_col = i;
    else if (header[i] == "right") right_col = i;
    else if (header[i] == "new_id") new_col = i;
  }
  if (step_col < 0 || left_col < 0 || right_col < 0 || new_col < 0) {
    throw std::runtime_error("merges csv must contain step,left,right,new_id columns");
  }

  std::vector<MergeRule> rules;
  int32_t seen = 0;
  while (static_cast<int32_t>(rules.size()) < limit && std::getline(in, line)) {
    if (line.empty()) continue;
    auto fields = split_csv_line(line);
    const int need = std::max(std::max(step_col, left_col), std::max(right_col, new_col));
    if (static_cast<int>(fields.size()) <= need) {
      throw std::runtime_error("malformed merges csv row: " + line);
    }
    if (seen++ < skip) continue;
    MergeRule rule;
    rule.step = std::stoi(fields[step_col]);
    rule.left = std::stoi(fields[left_col]);
    rule.right = std::stoi(fields[right_col]);
    rule.new_id = std::stoi(fields[new_col]);
    rules.push_back(rule);
  }
  if (static_cast<int32_t>(rules.size()) != limit) {
    throw std::runtime_error("requested " + std::to_string(limit) +
                             " merge rules but csv only provided " + std::to_string(rules.size()));
  }
  return rules;
}

uint64_t apply_rule(std::vector<int32_t> &seq, const MergeRule &rule, std::vector<int32_t> &buffer) {
  if (seq.size() < 2) return 0;
  buffer.clear();
  buffer.reserve(seq.size());
  uint64_t merges = 0;
  size_t i = 0;
  while (i < seq.size()) {
    if (i + 1 < seq.size() && seq[i] == rule.left && seq[i + 1] == rule.right) {
      buffer.push_back(rule.new_id);
      i += 2;
      merges += 1;
    } else {
      buffer.push_back(seq[i]);
      i += 1;
    }
  }
  seq.swap(buffer);
  return merges;
}

uint64_t pack_pair(int32_t left, int32_t right) {
  return (static_cast<uint64_t>(static_cast<uint32_t>(left)) << 32u) | static_cast<uint32_t>(right);
}

std::vector<int32_t> apply_rules_indexed(const std::vector<int32_t> &input,
                                         const std::vector<MergeRule> &rules,
                                         uint64_t &real_merges) {
  struct Node {
    int32_t val = 0;
    int prev = -1;
    int next = -1;
    bool alive = true;
  };
  struct Candidate {
    int step = 0;
    int pos = -1;
    int right = -1;
  };
  struct CandidateCmp {
    bool operator()(const Candidate &a, const Candidate &b) const {
      return a.step > b.step;
    }
  };

  const int n = static_cast<int>(input.size());
  if (n < 2 || rules.empty()) {
    return input;
  }

  std::vector<Node> nodes(n);
  for (int i = 0; i < n; ++i) {
    nodes[i].val = input[i];
    nodes[i].prev = i - 1;
    nodes[i].next = (i + 1 < n) ? i + 1 : -1;
  }

  std::unordered_map<uint64_t, int> pair_to_step;
  pair_to_step.reserve(rules.size() * 2 + 1);
  for (int i = 0; i < static_cast<int>(rules.size()); ++i) {
    pair_to_step.emplace(pack_pair(rules[i].left, rules[i].right), i);
  }

  std::priority_queue<Candidate, std::vector<Candidate>, CandidateCmp> pq;
  auto push_candidate = [&](int pos) {
    if (pos < 0 || pos >= n || !nodes[pos].alive) return;
    const int right = nodes[pos].next;
    if (right < 0 || !nodes[right].alive) return;
    auto it = pair_to_step.find(pack_pair(nodes[pos].val, nodes[right].val));
    if (it == pair_to_step.end()) return;
    pq.push({it->second, pos, right});
  };

  for (int i = 0; i + 1 < n; ++i) {
    push_candidate(i);
  }

  while (!pq.empty()) {
    Candidate cand = pq.top();
    pq.pop();
    if (cand.step < 0 || cand.step >= static_cast<int>(rules.size())) continue;
    const int pos = cand.pos;
    const int right = cand.right;
    if (pos < 0 || right < 0 || pos >= n || right >= n) continue;
    if (!nodes[pos].alive || !nodes[right].alive || nodes[pos].next != right) continue;
    const auto &rule = rules[cand.step];
    if (nodes[pos].val != rule.left || nodes[right].val != rule.right) continue;

    nodes[pos].val = rule.new_id;
    nodes[right].alive = false;
    const int after = nodes[right].next;
    nodes[pos].next = after;
    if (after >= 0) nodes[after].prev = pos;
    real_merges += 1;

    push_candidate(nodes[pos].prev);
    push_candidate(pos);
  }

  std::vector<int32_t> output;
  output.reserve(input.size());
  for (int i = 0; i >= 0; i = nodes[i].next) {
    if (nodes[i].alive) output.push_back(nodes[i].val);
  }
  return output;
}

std::string json_escape(const std::string &s) {
  std::string out;
  out.reserve(s.size() + 8);
  for (char c : s) {
    if (c == '"' || c == '\\') {
      out.push_back('\\');
      out.push_back(c);
    } else if (c == '\n') {
      out += "\\n";
    } else {
      out.push_back(c);
    }
  }
  return out;
}

void write_summary(const std::string &path,
                   const Args &args,
                   const std::vector<MergeRule> &rules,
                   const Summary &summary) {
  if (path.empty()) return;
  std::ofstream out(path);
  if (!out) throw std::runtime_error("failed to write summary: " + path);
  out << "{\n"
      << "  \"mode\": \"prefix_merge_int_shard\",\n"
      << "  \"input\": \"" << json_escape(args.input) << "\",\n"
      << "  \"output\": \"" << json_escape(args.output) << "\",\n"
      << "  \"merges_csv\": \"" << json_escape(args.merges_csv) << "\",\n"
      << "  \"skip\": " << args.skip << ",\n"
      << "  \"limit\": " << args.limit << ",\n"
      << "  \"rules_loaded\": " << rules.size() << ",\n"
      << "  \"input_records\": " << summary.input_records << ",\n"
      << "  \"input_tokens\": " << summary.input_tokens << ",\n"
      << "  \"output_records\": " << summary.output_records << ",\n"
      << "  \"output_tokens\": " << summary.output_tokens << ",\n"
      << "  \"compression_ratio\": "
      << (summary.output_tokens > 0 ? static_cast<double>(summary.input_tokens) / summary.output_tokens : 0.0) << ",\n"
      << "  \"real_merges\": " << summary.real_merges << ",\n"
      << "  \"input_max_token\": " << summary.input_max_token << ",\n"
      << "  \"output_max_token\": " << summary.output_max_token << ",\n"
      << "  \"elapsed_sec\": " << summary.elapsed_sec << "\n"
      << "}\n";
}

Args parse_args(int argc, char **argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    std::string key = argv[i];
    auto need_value = [&](const std::string &name) -> std::string {
      if (i + 1 >= argc) throw std::runtime_error("missing value for " + name);
      return argv[++i];
    };
    if (key == "--input") args.input = need_value(key);
    else if (key == "--output") args.output = need_value(key);
    else if (key == "--merges") args.merges_csv = need_value(key);
    else if (key == "--summary") args.summary = need_value(key);
    else if (key == "--skip") args.skip = std::stoi(need_value(key));
    else if (key == "--limit") args.limit = std::stoi(need_value(key));
    else if (key == "--log-every") args.log_every = static_cast<uint64_t>(std::stoull(need_value(key)));
    else if (key == "--overwrite") args.overwrite = true;
    else throw std::runtime_error("unknown argument: " + key);
  }
  if (args.input.empty()) throw std::runtime_error("--input is required");
  if (args.output.empty()) throw std::runtime_error("--output is required");
  if (args.merges_csv.empty()) throw std::runtime_error("--merges is required");
  if (args.skip < 0) throw std::runtime_error("--skip must be non-negative");
  if (args.limit < 0) throw std::runtime_error("--limit must be non-negative");
  return args;
}

}  // namespace

int main(int argc, char **argv) {
  try {
    Args args = parse_args(argc, argv);
    if (exists(args.output) && !args.overwrite) {
      throw std::runtime_error("output exists; pass --overwrite to replace: " + args.output);
    }
    const auto rules = read_merge_rules(args.merges_csv, args.skip, args.limit);
    const auto start = Clock::now();

    std::ifstream in(args.input, std::ios::binary);
    if (!in) throw std::runtime_error("failed to open input: " + args.input);
    const SampleMeta input_meta = read_header(in);

    const std::string tmp_output = args.output + ".tmp";
    std::ofstream out(tmp_output, std::ios::binary);
    if (!out) throw std::runtime_error("failed to open output tmp: " + tmp_output);
    SampleMeta output_meta;
    output_meta.max_token = input_meta.max_token;
    write_header(out, output_meta);

    Summary summary;
    summary.input_records = input_meta.nseq;
    summary.input_tokens = input_meta.ntokens;
    summary.input_max_token = input_meta.max_token;
    summary.output_max_token = input_meta.max_token;
    for (const auto &rule : rules) {
      summary.output_max_token = std::max(summary.output_max_token, rule.new_id);
    }

    std::vector<int32_t> seq;
    std::vector<int32_t> buffer;
    for (uint64_t rec = 0; rec < input_meta.nseq; ++rec) {
      uint32_t len = 0;
      read_raw(in, len);
      seq.resize(len);
      if (len > 0) {
        in.read(reinterpret_cast<char *>(seq.data()), static_cast<std::streamsize>(len * sizeof(int32_t)));
        if (!in) throw std::runtime_error("unexpected EOF in sequence tokens");
      }
      for (int32_t token : seq) {
        summary.output_max_token = std::max(summary.output_max_token, token);
      }
      seq = apply_rules_indexed(seq, rules, summary.real_merges);
      for (int32_t token : seq) {
        summary.output_max_token = std::max(summary.output_max_token, token);
      }

      const uint32_t out_len = static_cast<uint32_t>(seq.size());
      write_raw(out, out_len);
      if (out_len > 0) {
        out.write(reinterpret_cast<const char *>(seq.data()), static_cast<std::streamsize>(out_len * sizeof(int32_t)));
        if (!out) throw std::runtime_error("failed to write sequence tokens");
      }
      summary.output_records += 1;
      summary.output_tokens += out_len;
      if (args.log_every > 0 && summary.output_records % args.log_every == 0) {
        std::cerr << "prefix-merge progress records=" << summary.output_records
                  << " input_tokens=" << summary.input_tokens
                  << " output_tokens=" << summary.output_tokens
                  << " compression_ratio="
                  << (summary.output_tokens > 0 ? static_cast<double>(input_meta.ntokens) / summary.output_tokens : 0.0)
                  << " elapsed_sec=" << seconds_since(start) << std::endl;
      }
    }

    output_meta.nseq = summary.output_records;
    output_meta.ntokens = summary.output_tokens;
    output_meta.max_token = summary.output_max_token;
    write_header(out, output_meta);
    out.close();
    if (!out) throw std::runtime_error("failed to finalize output: " + tmp_output);
    if (std::rename(tmp_output.c_str(), args.output.c_str()) != 0) {
      throw std::runtime_error("failed to move tmp output into place");
    }

    summary.elapsed_sec = seconds_since(start);
    write_summary(args.summary, args, rules, summary);
    std::cerr << "prefix-merge done records=" << summary.output_records
              << " input_tokens=" << summary.input_tokens
              << " output_tokens=" << summary.output_tokens
              << " compression_ratio="
              << (summary.output_tokens > 0 ? static_cast<double>(summary.input_tokens) / summary.output_tokens : 0.0)
              << " max_token=" << summary.output_max_token
              << " elapsed_sec=" << summary.elapsed_sec << std::endl;
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "ERROR: " << e.what() << std::endl;
    return 1;
  }
}
