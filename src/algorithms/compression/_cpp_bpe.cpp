#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cstdint>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <array>
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>

namespace py = pybind11;

namespace {

using PairKey = uint64_t;
using PairCount = int64_t;
using Clock = std::chrono::steady_clock;

inline int64_t elapsed_ns(Clock::time_point start) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start).count();
}

struct TimingProfile {
    bool enabled = false;
    int32_t num_threads = 0;
    int32_t completed_merges = 0;
    uint64_t input_sequences = 0;
    uint64_t input_tokens = 0;
    uint64_t nonempty_threads = 0;
    uint64_t range_tokens_min = 0;
    uint64_t range_tokens_max = 0;
    uint64_t local_count_workspace_mb = 0;
    uint64_t global_count_workspace_mb = 0;
    int64_t base_vocab_scan_ns = 0;
    int64_t trainer_setup_ns = 0;
    int64_t merge_loop_ns = 0;
    uint64_t count_rounds = 0;
    uint64_t apply_rounds = 0;
    uint64_t count_pairs_visited = 0;
    uint64_t local_touched_total = 0;
    uint64_t global_unique_total = 0;
    uint64_t apply_tokens_scanned = 0;
    uint64_t apply_sequences_scanned = 0;
    uint64_t apply_sequences_changed = 0;
    int64_t count_wall_ns = 0;
    int64_t count_start_delay_sum_ns = 0;
    int64_t count_start_delay_max_ns = 0;
    int64_t count_active_span_max_ns = 0;
    int64_t count_tail_after_active_ns = 0;
    int64_t count_worker_sum_ns = 0;
    int64_t count_worker_max_ns = 0;
    int64_t reduce_ns = 0;
    int64_t select_ns = 0;
    int64_t clear_ns = 0;
    int64_t apply_wall_ns = 0;
    int64_t apply_start_delay_sum_ns = 0;
    int64_t apply_start_delay_max_ns = 0;
    int64_t apply_active_span_max_ns = 0;
    int64_t apply_tail_after_active_ns = 0;
    int64_t apply_worker_sum_ns = 0;
    int64_t apply_worker_max_ns = 0;
};

struct TrainResult {
    std::vector<std::array<int32_t, 3>> merges;
    int32_t base_vocab_size = 0;
    int32_t final_vocab_size = 0;
    int32_t separator_token = 0;
    TimingProfile profile;
};

struct PairCoord {
    int32_t left = 0;
    int32_t right = 0;
};

inline PairKey make_pair_key(int32_t left, int32_t right) {
    return (static_cast<PairKey>(static_cast<uint32_t>(left)) << 32)
           | static_cast<uint32_t>(right);
}

inline std::pair<int32_t, int32_t> unpack_pair_key(PairKey key) {
    return {
        static_cast<int32_t>((key >> 32) & UINT64_C(0xFFFFFFFF)),
        static_cast<int32_t>(key & UINT64_C(0xFFFFFFFF)),
    };
}

inline bool pair_is_better(
    PairCount freq,
    int32_t left,
    int32_t right,
    PairCount best_freq,
    int32_t best_left,
    int32_t best_right
) {
    return freq > best_freq
           || (freq == best_freq
               && (left < best_left || (left == best_left && right < best_right)));
}

std::unordered_map<PairKey, PairCount> count_pairs_serial(
    const std::vector<int32_t> &ids,
    int32_t separator
) {
    std::unordered_map<PairKey, PairCount> counts;
    counts.reserve(ids.size() / 2 + 1);
    const size_t n = ids.size();
    for (size_t i = 0; i + 1 < n; ++i) {
        int32_t left = ids[i];
        int32_t right = ids[i + 1];
        if (left == separator || right == separator) continue;
        PairKey key = make_pair_key(left, right);
        auto it = counts.find(key);
        if (it == counts.end()) counts.emplace(key, 1);
        else it->second += 1;
    }
    return counts;
}

std::unordered_map<PairKey, PairCount> count_pairs_parallel(
    const std::vector<int32_t> &ids,
    int32_t separator,
    int32_t requested_threads
) {
    const size_t pair_count = ids.size() > 1 ? ids.size() - 1 : 0;
    if (pair_count < 1000000) {
        return count_pairs_serial(ids, separator);
    }

    unsigned int hardware_threads = std::thread::hardware_concurrency();
    if (hardware_threads == 0) hardware_threads = 1;
    size_t num_threads = requested_threads > 0
        ? static_cast<size_t>(requested_threads)
        : static_cast<size_t>(hardware_threads);
    num_threads = std::max<size_t>(1, std::min(num_threads, pair_count));
    if (num_threads <= 1) {
        return count_pairs_serial(ids, separator);
    }

    std::vector<std::unordered_map<PairKey, PairCount>> local_counts(num_threads);
    std::vector<std::thread> workers;
    workers.reserve(num_threads);
    const size_t chunk = (pair_count + num_threads - 1) / num_threads;

    for (size_t tid = 0; tid < num_threads; ++tid) {
        const size_t begin = tid * chunk;
        const size_t end = std::min(pair_count, begin + chunk);
        workers.emplace_back([&, tid, begin, end]() {
            auto &counts = local_counts[tid];
            counts.reserve(std::min<size_t>((end - begin) / 4 + 1, 1000000));
            for (size_t i = begin; i < end; ++i) {
                int32_t left = ids[i];
                int32_t right = ids[i + 1];
                if (left == separator || right == separator) continue;
                PairKey key = make_pair_key(left, right);
                auto it = counts.find(key);
                if (it == counts.end()) counts.emplace(key, 1);
                else it->second += 1;
            }
        });
    }
    for (auto &worker : workers) {
        worker.join();
    }

    size_t total_local_size = 0;
    for (const auto &counts : local_counts) {
        total_local_size += counts.size();
    }
    std::unordered_map<PairKey, PairCount> counts;
    counts.reserve(total_local_size + 1);
    for (const auto &local : local_counts) {
        for (const auto &kv : local) {
            auto it = counts.find(kv.first);
            if (it == counts.end()) counts.emplace(kv.first, kv.second);
            else it->second += kv.second;
        }
    }
    return counts;
}

TrainResult train_minbpe_impl(
    const std::vector<std::vector<int32_t>> &seqs,
    int32_t num_merges,
    int32_t min_frequency,
    bool parallel_count,
    int32_t num_threads
) {
    std::unordered_set<int32_t> base_vocab;
    base_vocab.reserve(1024);
    int32_t max_base_id = -1;
    for (const auto &s : seqs) {
        for (int32_t t : s) {
            base_vocab.insert(t);
            if (t > max_base_id) max_base_id = t;
        }
    }
    const int32_t separator = max_base_id + 1;
    int32_t next_id = separator + 1;

    std::vector<int32_t> ids;
    size_t total_len = 0;
    for (const auto &s : seqs) total_len += s.size();
    if (!seqs.empty()) total_len += (seqs.size() - 1);
    ids.reserve(total_len);
    for (size_t i = 0; i < seqs.size(); ++i) {
        if (i > 0) ids.push_back(separator);
        const auto &s = seqs[i];
        ids.insert(ids.end(), s.begin(), s.end());
    }

    std::vector<std::array<int32_t, 3>> merges;
    merges.reserve(static_cast<size_t>(num_merges));

    for (int32_t step = 0; step < num_merges; ++step) {
        std::unordered_map<PairKey, PairCount> counts = parallel_count
            ? count_pairs_parallel(ids, separator, num_threads)
            : count_pairs_serial(ids, separator);

        PairCount best_freq = -1;
        int32_t best_left = 0;
        int32_t best_right = 0;
        for (const auto &kv : counts) {
            PairCount freq = kv.second;
            if (freq < min_frequency) continue;
            auto pair = unpack_pair_key(kv.first);
            int32_t left = pair.first;
            int32_t right = pair.second;
            if (pair_is_better(freq, left, right, best_freq, best_left, best_right)) {
                best_freq = freq;
                best_left = left;
                best_right = right;
            }
        }
        if (best_freq < min_frequency) {
            break;
        }

        const int32_t new_id = next_id++;
        std::vector<int32_t> out;
        out.reserve(ids.size());
        for (size_t i = 0; i < ids.size(); ) {
            if (i + 1 < ids.size() && ids[i] == best_left && ids[i + 1] == best_right) {
                out.push_back(new_id);
                i += 2;
            } else {
                out.push_back(ids[i]);
                i += 1;
            }
        }
        ids.swap(out);
        merges.push_back({best_left, best_right, new_id});
    }

    TrainResult result;
    result.merges = std::move(merges);
    result.base_vocab_size = static_cast<int32_t>(base_vocab.size());
    result.final_vocab_size = result.base_vocab_size + static_cast<int32_t>(result.merges.size());
    result.separator_token = separator;
    return result;
}

void apply_merge_serial_reuse(
    const std::vector<int32_t> &ids,
    int32_t best_left,
    int32_t best_right,
    int32_t new_id,
    std::vector<int32_t> &out
) {
    out.clear();
    out.reserve(ids.size());
    for (size_t i = 0; i < ids.size(); ) {
        if (i + 1 < ids.size() && ids[i] == best_left && ids[i + 1] == best_right) {
            out.push_back(new_id);
            i += 2;
        } else {
            out.push_back(ids[i]);
            i += 1;
        }
    }
}

std::vector<int32_t> apply_merge_parallel_distinct(
    const std::vector<int32_t> &ids,
    int32_t best_left,
    int32_t best_right,
    int32_t new_id,
    size_t num_threads
) {
    const size_t n = ids.size();
    const size_t min_parallel_ids = 1000000;
    if (best_left == best_right || num_threads <= 1 || n < min_parallel_ids) {
        std::vector<int32_t> out;
        apply_merge_serial_reuse(ids, best_left, best_right, new_id, out);
        return out;
    }

    const size_t active_threads = std::max<size_t>(1, std::min(num_threads, n));
    std::vector<size_t> local_sizes(active_threads, 0);
    std::vector<std::thread> workers;
    workers.reserve(active_threads);

    auto range_begin = [n, active_threads](size_t tid) -> size_t {
        return (n * tid) / active_threads;
    };
    auto range_end = [n, active_threads](size_t tid) -> size_t {
        return (n * (tid + 1)) / active_threads;
    };
    auto skip_boundary_right = [&](size_t begin) -> bool {
        return begin > 0 && begin < n
               && ids[begin - 1] == best_left
               && ids[begin] == best_right;
    };

    for (size_t tid = 0; tid < active_threads; ++tid) {
        const size_t begin = range_begin(tid);
        const size_t end = range_end(tid);
        workers.emplace_back([&, tid, begin, end]() {
            size_t i = skip_boundary_right(begin) ? begin + 1 : begin;
            size_t out_len = 0;
            while (i < end) {
                if (i + 1 < n && ids[i] == best_left && ids[i + 1] == best_right) {
                    out_len += 1;
                    i += 2;
                } else {
                    out_len += 1;
                    i += 1;
                }
            }
            local_sizes[tid] = out_len;
        });
    }
    for (auto &worker : workers) {
        worker.join();
    }

    std::vector<size_t> offsets(active_threads + 1, 0);
    for (size_t tid = 0; tid < active_threads; ++tid) {
        offsets[tid + 1] = offsets[tid] + local_sizes[tid];
    }
    std::vector<int32_t> out(offsets.back());

    workers.clear();
    for (size_t tid = 0; tid < active_threads; ++tid) {
        const size_t begin = range_begin(tid);
        const size_t end = range_end(tid);
        const size_t out_begin = offsets[tid];
        workers.emplace_back([&, begin, end, out_begin]() {
            size_t i = skip_boundary_right(begin) ? begin + 1 : begin;
            size_t pos = out_begin;
            while (i < end) {
                if (i + 1 < n && ids[i] == best_left && ids[i + 1] == best_right) {
                    out[pos++] = new_id;
                    i += 2;
                } else {
                    out[pos++] = ids[i];
                    i += 1;
                }
            }
        });
    }
    for (auto &worker : workers) {
        worker.join();
    }
    return out;
}

class DensePairCounter {
public:
    DensePairCounter(size_t pair_dim, size_t num_threads)
        : pair_dim_(pair_dim),
          pair_space_(pair_dim * pair_dim),
          num_threads_(num_threads),
          local_counts_(num_threads),
          local_touched_(num_threads),
          global_counts_(pair_space_, 0)
    {
        if (pair_dim_ == 0 || pair_dim_ > std::numeric_limits<uint32_t>::max() / pair_dim_) {
            throw std::runtime_error("dense BPE pair space overflow");
        }
        if (pair_space_ > std::numeric_limits<uint32_t>::max()) {
            throw std::runtime_error("dense BPE pair space exceeds uint32 key range");
        }
        const size_t local_bytes = pair_space_ * sizeof(uint32_t) * num_threads_;
        const size_t global_bytes = pair_space_ * sizeof(PairCount);
        const size_t max_workspace_bytes = static_cast<size_t>(64) * 1024 * 1024 * 1024;
        if (local_bytes + global_bytes > max_workspace_bytes) {
            throw std::runtime_error("dense BPE workspace would exceed 64GB");
        }
        for (size_t tid = 0; tid < num_threads_; ++tid) {
            local_counts_[tid].assign(pair_space_, 0);
            local_touched_[tid].reserve(1024 * 1024);
        }
        global_touched_.reserve(1024 * 1024);
    }

    void count(
        const std::vector<int32_t> &ids,
        int32_t separator,
        PairCount min_frequency,
        PairCount &best_freq,
        int32_t &best_left,
        int32_t &best_right
    ) {
        const size_t pair_count = ids.size() > 1 ? ids.size() - 1 : 0;
        const size_t active_threads = std::max<size_t>(
            1,
            std::min(num_threads_, pair_count == 0 ? static_cast<size_t>(1) : pair_count)
        );
        const size_t chunk = active_threads > 0 ? (pair_count + active_threads - 1) / active_threads : 0;
        std::vector<std::thread> workers;
        workers.reserve(active_threads);

        for (size_t tid = 0; tid < active_threads; ++tid) {
            const size_t begin = tid * chunk;
            const size_t end = std::min(pair_count, begin + chunk);
            workers.emplace_back([&, tid, begin, end]() {
                auto &counts = local_counts_[tid];
                auto &touched = local_touched_[tid];
                for (size_t i = begin; i < end; ++i) {
                    int32_t left = ids[i];
                    int32_t right = ids[i + 1];
                    if (left == separator || right == separator) continue;
                    const uint32_t key = static_cast<uint32_t>(
                        static_cast<size_t>(left) * pair_dim_ + static_cast<size_t>(right)
                    );
                    if (counts[key] == 0) {
                        touched.push_back(key);
                    }
                    counts[key] += 1;
                }
            });
        }
        for (auto &worker : workers) {
            worker.join();
        }

        for (size_t tid = 0; tid < active_threads; ++tid) {
            const auto &touched = local_touched_[tid];
            const auto &counts = local_counts_[tid];
            for (uint32_t key : touched) {
                const PairCount value = static_cast<PairCount>(counts[key]);
                if (global_counts_[key] == 0) {
                    global_touched_.push_back(key);
                }
                global_counts_[key] += value;
            }
        }

        best_freq = -1;
        best_left = 0;
        best_right = 0;
        for (uint32_t key : global_touched_) {
            PairCount freq = global_counts_[key];
            if (freq < min_frequency) continue;
            int32_t left = static_cast<int32_t>(key / pair_dim_);
            int32_t right = static_cast<int32_t>(key % pair_dim_);
            if (pair_is_better(freq, left, right, best_freq, best_left, best_right)) {
                best_freq = freq;
                best_left = left;
                best_right = right;
            }
        }

        clear(active_threads);
    }

private:
    void clear(size_t active_threads) {
        for (uint32_t key : global_touched_) {
            global_counts_[key] = 0;
        }
        global_touched_.clear();

        for (size_t tid = 0; tid < active_threads; ++tid) {
            auto &counts = local_counts_[tid];
            auto &touched = local_touched_[tid];
            for (uint32_t key : touched) {
                counts[key] = 0;
            }
            touched.clear();
        }
    }

    size_t pair_dim_;
    size_t pair_space_;
    size_t num_threads_;
    std::vector<std::vector<uint32_t>> local_counts_;
    std::vector<std::vector<uint32_t>> local_touched_;
    std::vector<PairCount> global_counts_;
    std::vector<uint32_t> global_touched_;
};

class DensePairCounterPool {
public:
    DensePairCounterPool(size_t pair_dim, size_t num_threads)
        : pair_dim_(pair_dim),
          pair_space_(pair_dim * pair_dim),
          num_threads_(num_threads),
          local_counts_(num_threads),
          local_touched_(num_threads),
          global_counts_(pair_space_, 0)
    {
        if (pair_dim_ == 0 || pair_dim_ > std::numeric_limits<uint32_t>::max() / pair_dim_) {
            throw std::runtime_error("dense BPE pair space overflow");
        }
        if (pair_space_ > std::numeric_limits<uint32_t>::max()) {
            throw std::runtime_error("dense BPE pair space exceeds uint32 key range");
        }
        const size_t local_bytes = pair_space_ * sizeof(uint32_t) * num_threads_;
        const size_t global_bytes = pair_space_ * sizeof(PairCount);
        const size_t max_workspace_bytes = static_cast<size_t>(64) * 1024 * 1024 * 1024;
        if (local_bytes + global_bytes > max_workspace_bytes) {
            throw std::runtime_error("dense BPE workspace would exceed 64GB");
        }
        for (size_t tid = 0; tid < num_threads_; ++tid) {
            local_counts_[tid].assign(pair_space_, 0);
            local_touched_[tid].reserve(1024 * 1024);
        }
        global_touched_.reserve(1024 * 1024);
        workers_.reserve(num_threads_);
        for (size_t tid = 0; tid < num_threads_; ++tid) {
            workers_.emplace_back([this, tid]() { worker_loop(tid); });
        }
    }

    ~DensePairCounterPool() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_ = true;
            work_generation_ += 1;
        }
        cv_start_.notify_all();
        for (auto &worker : workers_) {
            if (worker.joinable()) {
                worker.join();
            }
        }
    }

    DensePairCounterPool(const DensePairCounterPool &) = delete;
    DensePairCounterPool &operator=(const DensePairCounterPool &) = delete;

    void count(
        const std::vector<int32_t> &ids,
        int32_t separator,
        PairCount min_frequency,
        PairCount &best_freq,
        int32_t &best_left,
        int32_t &best_right
    ) {
        const size_t pair_count = ids.size() > 1 ? ids.size() - 1 : 0;
        const size_t active_threads = std::max<size_t>(
            1,
            std::min(num_threads_, pair_count == 0 ? static_cast<size_t>(1) : pair_count)
        );
        const size_t chunk = active_threads > 0 ? (pair_count + active_threads - 1) / active_threads : 0;

        {
            std::lock_guard<std::mutex> lock(mutex_);
            ids_ = &ids;
            separator_ = separator;
            pair_count_ = pair_count;
            active_threads_ = active_threads;
            chunk_ = chunk;
            completed_threads_ = 0;
            work_generation_ += 1;
        }
        cv_start_.notify_all();
        {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_done_.wait(lock, [&]() {
                return completed_threads_ >= active_threads_;
            });
        }

        for (size_t tid = 0; tid < active_threads; ++tid) {
            const auto &touched = local_touched_[tid];
            const auto &counts = local_counts_[tid];
            for (uint32_t key : touched) {
                const PairCount value = static_cast<PairCount>(counts[key]);
                if (global_counts_[key] == 0) {
                    global_touched_.push_back(key);
                }
                global_counts_[key] += value;
            }
        }

        best_freq = -1;
        best_left = 0;
        best_right = 0;
        for (uint32_t key : global_touched_) {
            PairCount freq = global_counts_[key];
            if (freq < min_frequency) continue;
            int32_t left = static_cast<int32_t>(key / pair_dim_);
            int32_t right = static_cast<int32_t>(key % pair_dim_);
            if (pair_is_better(freq, left, right, best_freq, best_left, best_right)) {
                best_freq = freq;
                best_left = left;
                best_right = right;
            }
        }

        clear(active_threads);
    }

private:
    void worker_loop(size_t tid) {
        size_t seen_generation = 0;
        while (true) {
            const std::vector<int32_t> *ids = nullptr;
            int32_t separator = 0;
            size_t pair_count = 0;
            size_t begin = 0;
            size_t end = 0;
            bool active = false;

            {
                std::unique_lock<std::mutex> lock(mutex_);
                cv_start_.wait(lock, [&]() {
                    return stop_ || work_generation_ != seen_generation;
                });
                if (stop_) {
                    return;
                }
                seen_generation = work_generation_;
                active = tid < active_threads_;
                if (active) {
                    ids = ids_;
                    separator = separator_;
                    pair_count = pair_count_;
                    begin = tid * chunk_;
                    end = std::min(pair_count, begin + chunk_);
                }
            }

            if (active) {
                auto &counts = local_counts_[tid];
                auto &touched = local_touched_[tid];
                for (size_t i = begin; i < end; ++i) {
                    int32_t left = (*ids)[i];
                    int32_t right = (*ids)[i + 1];
                    if (left == separator || right == separator) continue;
                    const uint32_t key = static_cast<uint32_t>(
                        static_cast<size_t>(left) * pair_dim_ + static_cast<size_t>(right)
                    );
                    if (counts[key] == 0) {
                        touched.push_back(key);
                    }
                    counts[key] += 1;
                }
                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    completed_threads_ += 1;
                    if (completed_threads_ >= active_threads_) {
                        cv_done_.notify_one();
                    }
                }
            }
        }
    }

    void clear(size_t active_threads) {
        for (uint32_t key : global_touched_) {
            global_counts_[key] = 0;
        }
        global_touched_.clear();

        for (size_t tid = 0; tid < active_threads; ++tid) {
            auto &counts = local_counts_[tid];
            auto &touched = local_touched_[tid];
            for (uint32_t key : touched) {
                counts[key] = 0;
            }
            touched.clear();
        }
    }

    size_t pair_dim_;
    size_t pair_space_;
    size_t num_threads_;
    std::vector<std::vector<uint32_t>> local_counts_;
    std::vector<std::vector<uint32_t>> local_touched_;
    std::vector<PairCount> global_counts_;
    std::vector<uint32_t> global_touched_;
    std::vector<std::thread> workers_;
    std::mutex mutex_;
    std::condition_variable cv_start_;
    std::condition_variable cv_done_;
    bool stop_ = false;
    size_t work_generation_ = 0;
    size_t completed_threads_ = 0;
    const std::vector<int32_t> *ids_ = nullptr;
    int32_t separator_ = 0;
    size_t pair_count_ = 0;
    size_t active_threads_ = 1;
    size_t chunk_ = 0;
};

class DenseSampleParallelTrainer {
public:
    DenseSampleParallelTrainer(
        std::vector<std::vector<int32_t>> seqs,
        size_t pair_dim,
        size_t num_threads
    )
        : seqs_(std::move(seqs)),
          pair_dim_(pair_dim),
          pair_space_(pair_dim * pair_dim),
          num_threads_(std::max<size_t>(1, std::min(num_threads, std::max<size_t>(1, seqs_.size())))),
          ranges_(make_ranges()),
          local_counts_(num_threads_),
          local_touched_(num_threads_),
          apply_buffers_(num_threads_),
          global_counts_(pair_space_, 0)
    {
        if (pair_dim_ == 0 || pair_dim_ > std::numeric_limits<uint32_t>::max() / pair_dim_) {
            throw std::runtime_error("dense sample-parallel BPE pair space overflow");
        }
        if (pair_space_ > std::numeric_limits<uint32_t>::max()) {
            throw std::runtime_error("dense sample-parallel BPE pair space exceeds uint32 key range");
        }
        const size_t local_bytes = pair_space_ * sizeof(uint32_t) * num_threads_;
        const size_t global_bytes = pair_space_ * sizeof(PairCount);
        const size_t max_workspace_bytes = static_cast<size_t>(64) * 1024 * 1024 * 1024;
        if (local_bytes + global_bytes > max_workspace_bytes) {
            throw std::runtime_error("dense sample-parallel BPE workspace would exceed 64GB");
        }
        for (size_t tid = 0; tid < num_threads_; ++tid) {
            local_counts_[tid].assign(pair_space_, 0);
            local_touched_[tid].reserve(1024 * 1024);
        }
        global_touched_.reserve(1024 * 1024);
        workers_.reserve(num_threads_);
        for (size_t tid = 0; tid < num_threads_; ++tid) {
            workers_.emplace_back([this, tid]() { worker_loop(tid); });
        }
    }

    ~DenseSampleParallelTrainer() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_ = true;
            work_generation_ += 1;
        }
        cv_start_.notify_all();
        for (auto &worker : workers_) {
            if (worker.joinable()) {
                worker.join();
            }
        }
    }

    DenseSampleParallelTrainer(const DenseSampleParallelTrainer &) = delete;
    DenseSampleParallelTrainer &operator=(const DenseSampleParallelTrainer &) = delete;

    void count_best_pair(
        PairCount min_frequency,
        PairCount &best_freq,
        int32_t &best_left,
        int32_t &best_right,
        TimingProfile *profile = nullptr
    ) {
        const auto count_start = Clock::now();
        run_task(Task::Count, 0, 0, 0);
        if (profile != nullptr && profile->enabled) {
            profile->count_rounds += 1;
            const int64_t wall_ns = elapsed_ns(count_start);
            profile->count_wall_ns += wall_ns;
            uint64_t pairs = 0;
            uint64_t touched = 0;
            int64_t max_worker_ns = 0;
            int64_t max_start_delay_ns = 0;
            int64_t sum_start_delay_ns = 0;
            int64_t max_finish_ns = 0;
            for (size_t tid = 0; tid < num_threads_; ++tid) {
                pairs += worker_pair_visits_[tid];
                touched += static_cast<uint64_t>(local_touched_[tid].size());
                profile->count_worker_sum_ns += worker_elapsed_ns_[tid];
                if (worker_elapsed_ns_[tid] > max_worker_ns) {
                    max_worker_ns = worker_elapsed_ns_[tid];
                }
                const int64_t start_delay_ns = worker_start_delay_ns_[tid];
                sum_start_delay_ns += start_delay_ns;
                if (start_delay_ns > max_start_delay_ns) {
                    max_start_delay_ns = start_delay_ns;
                }
                const int64_t finish_ns = start_delay_ns + worker_elapsed_ns_[tid];
                if (finish_ns > max_finish_ns) {
                    max_finish_ns = finish_ns;
                }
            }
            profile->count_pairs_visited += pairs;
            profile->local_touched_total += touched;
            profile->count_worker_max_ns += max_worker_ns;
            profile->count_start_delay_sum_ns += sum_start_delay_ns;
            profile->count_start_delay_max_ns += max_start_delay_ns;
            profile->count_active_span_max_ns += max_finish_ns;
            profile->count_tail_after_active_ns += wall_ns > max_finish_ns ? wall_ns - max_finish_ns : 0;
        }

        const auto reduce_start = Clock::now();
        for (size_t tid = 0; tid < num_threads_; ++tid) {
            const auto &touched = local_touched_[tid];
            const auto &counts = local_counts_[tid];
            for (uint32_t key : touched) {
                const PairCount value = static_cast<PairCount>(counts[key]);
                if (global_counts_[key] == 0) {
                    global_touched_.push_back(key);
                }
                global_counts_[key] += value;
            }
        }
        if (profile != nullptr && profile->enabled) {
            profile->reduce_ns += elapsed_ns(reduce_start);
            profile->global_unique_total += static_cast<uint64_t>(global_touched_.size());
        }

        const auto select_start = Clock::now();
        best_freq = -1;
        best_left = 0;
        best_right = 0;
        for (uint32_t key : global_touched_) {
            PairCount freq = global_counts_[key];
            if (freq < min_frequency) continue;
            int32_t left = static_cast<int32_t>(key / pair_dim_);
            int32_t right = static_cast<int32_t>(key % pair_dim_);
            if (pair_is_better(freq, left, right, best_freq, best_left, best_right)) {
                best_freq = freq;
                best_left = left;
                best_right = right;
            }
        }
        if (profile != nullptr && profile->enabled) {
            profile->select_ns += elapsed_ns(select_start);
        }

        const auto clear_start = Clock::now();
        clear_counts();
        if (profile != nullptr && profile->enabled) {
            profile->clear_ns += elapsed_ns(clear_start);
        }
    }

    void apply_merge(int32_t best_left, int32_t best_right, int32_t new_id, TimingProfile *profile = nullptr) {
        const auto apply_start = Clock::now();
        run_task(Task::Apply, best_left, best_right, new_id);
        if (profile != nullptr && profile->enabled) {
            profile->apply_rounds += 1;
            const int64_t wall_ns = elapsed_ns(apply_start);
            profile->apply_wall_ns += wall_ns;
            int64_t max_worker_ns = 0;
            int64_t max_start_delay_ns = 0;
            int64_t sum_start_delay_ns = 0;
            int64_t max_finish_ns = 0;
            for (size_t tid = 0; tid < num_threads_; ++tid) {
                profile->apply_worker_sum_ns += worker_elapsed_ns_[tid];
                if (worker_elapsed_ns_[tid] > max_worker_ns) {
                    max_worker_ns = worker_elapsed_ns_[tid];
                }
                const int64_t start_delay_ns = worker_start_delay_ns_[tid];
                sum_start_delay_ns += start_delay_ns;
                if (start_delay_ns > max_start_delay_ns) {
                    max_start_delay_ns = start_delay_ns;
                }
                const int64_t finish_ns = start_delay_ns + worker_elapsed_ns_[tid];
                if (finish_ns > max_finish_ns) {
                    max_finish_ns = finish_ns;
                }
                profile->apply_tokens_scanned += worker_apply_tokens_[tid];
                profile->apply_sequences_scanned += worker_apply_sequences_[tid];
                profile->apply_sequences_changed += worker_apply_changed_[tid];
            }
            profile->apply_worker_max_ns += max_worker_ns;
            profile->apply_start_delay_sum_ns += sum_start_delay_ns;
            profile->apply_start_delay_max_ns += max_start_delay_ns;
            profile->apply_active_span_max_ns += max_finish_ns;
            profile->apply_tail_after_active_ns += wall_ns > max_finish_ns ? wall_ns - max_finish_ns : 0;
        }
    }

    void fill_static_profile(TimingProfile &profile) const {
        profile.num_threads = static_cast<int32_t>(num_threads_);
        profile.input_sequences = static_cast<uint64_t>(seqs_.size());
        uint64_t min_tokens = std::numeric_limits<uint64_t>::max();
        uint64_t max_tokens = 0;
        uint64_t nonempty = 0;
        uint64_t total_tokens = 0;
        for (size_t tid = 0; tid < num_threads_; ++tid) {
            uint64_t range_tokens = 0;
            const auto range = ranges_[tid];
            for (size_t si = range.first; si < range.second; ++si) {
                range_tokens += static_cast<uint64_t>(seqs_[si].size());
            }
            if (range.second > range.first) {
                nonempty += 1;
            }
            min_tokens = std::min(min_tokens, range_tokens);
            max_tokens = std::max(max_tokens, range_tokens);
            total_tokens += range_tokens;
        }
        if (num_threads_ == 0) {
            min_tokens = 0;
        }
        profile.input_tokens = total_tokens;
        profile.nonempty_threads = nonempty;
        profile.range_tokens_min = min_tokens == std::numeric_limits<uint64_t>::max() ? 0 : min_tokens;
        profile.range_tokens_max = max_tokens;
        profile.local_count_workspace_mb = static_cast<uint64_t>(
            (static_cast<long double>(pair_space_) * sizeof(uint32_t) * num_threads_) / (1024.0L * 1024.0L)
        );
        profile.global_count_workspace_mb = static_cast<uint64_t>(
            (static_cast<long double>(pair_space_) * sizeof(PairCount)) / (1024.0L * 1024.0L)
        );
    }

private:
    enum class Task {
        Count,
        Apply,
    };

    std::vector<std::pair<size_t, size_t>> make_ranges() const {
        std::vector<std::pair<size_t, size_t>> ranges;
        ranges.reserve(num_threads_);
        const size_t nseq = seqs_.size();
        if (nseq == 0) {
            ranges.assign(num_threads_, {0, 0});
            return ranges;
        }

        size_t total_tokens = 0;
        for (const auto &seq : seqs_) {
            total_tokens += seq.size();
        }

        size_t start = 0;
        size_t consumed_tokens = 0;
        for (size_t tid = 0; tid < num_threads_; ++tid) {
            if (start >= nseq) {
                ranges.push_back({nseq, nseq});
                continue;
            }
            if (tid + 1 == num_threads_) {
                ranges.push_back({start, nseq});
                break;
            }
            const size_t remaining_threads = num_threads_ - tid;
            const size_t remaining_tokens = total_tokens > consumed_tokens
                ? total_tokens - consumed_tokens
                : 0;
            const size_t target = std::max<size_t>(1, (remaining_tokens + remaining_threads - 1) / remaining_threads);
            size_t end = start;
            size_t cur_tokens = 0;
            while (end < nseq && (nseq - end) > (remaining_threads - 1)) {
                if (end > start && cur_tokens >= target) {
                    break;
                }
                cur_tokens += seqs_[end].size();
                ++end;
            }
            ranges.push_back({start, end});
            start = end;
            consumed_tokens += cur_tokens;
        }
        while (ranges.size() < num_threads_) {
            ranges.push_back({nseq, nseq});
        }
        return ranges;
    }

    void run_task(Task task, int32_t best_left, int32_t best_right, int32_t new_id) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            task_ = task;
            best_left_ = best_left;
            best_right_ = best_right;
            new_id_ = new_id;
            completed_threads_ = 0;
            dispatch_time_ = Clock::now();
            work_generation_ += 1;
        }
        cv_start_.notify_all();
        {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_done_.wait(lock, [&]() {
                return completed_threads_ >= num_threads_;
            });
        }
    }

    void worker_loop(size_t tid) {
        size_t seen_generation = 0;
        while (true) {
            Task task = Task::Count;
            int32_t best_left = 0;
            int32_t best_right = 0;
            int32_t new_id = 0;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                cv_start_.wait(lock, [&]() {
                    return stop_ || work_generation_ != seen_generation;
                });
                if (stop_) {
                    return;
                }
                seen_generation = work_generation_;
                task = task_;
                best_left = best_left_;
                best_right = best_right_;
                new_id = new_id_;
            }

            const auto task_start = Clock::now();
            worker_start_delay_ns_[tid] = std::chrono::duration_cast<std::chrono::nanoseconds>(
                task_start - dispatch_time_
            ).count();
            if (task == Task::Count) {
                count_range(tid);
            } else {
                apply_range(tid, best_left, best_right, new_id);
            }
            worker_elapsed_ns_[tid] = elapsed_ns(task_start);

            {
                std::lock_guard<std::mutex> lock(mutex_);
                completed_threads_ += 1;
                if (completed_threads_ >= num_threads_) {
                    cv_done_.notify_one();
                }
            }
        }
    }

    void count_range(size_t tid) {
        auto &counts = local_counts_[tid];
        auto &touched = local_touched_[tid];
        const auto range = ranges_[tid];
        uint64_t pair_visits = 0;
        for (size_t si = range.first; si < range.second; ++si) {
            const auto &seq = seqs_[si];
            for (size_t i = 0; i + 1 < seq.size(); ++i) {
                const int32_t left = seq[i];
                const int32_t right = seq[i + 1];
                const uint32_t key = static_cast<uint32_t>(
                    static_cast<size_t>(left) * pair_dim_ + static_cast<size_t>(right)
                );
                if (counts[key] == 0) {
                    touched.push_back(key);
                }
                counts[key] += 1;
                pair_visits += 1;
            }
        }
        worker_pair_visits_[tid] = pair_visits;
        worker_apply_tokens_[tid] = 0;
        worker_apply_sequences_[tid] = 0;
        worker_apply_changed_[tid] = 0;
    }

    void apply_range(size_t tid, int32_t best_left, int32_t best_right, int32_t new_id) {
        auto &buffer = apply_buffers_[tid];
        const auto range = ranges_[tid];
        uint64_t tokens_scanned = 0;
        uint64_t sequences_scanned = 0;
        uint64_t sequences_changed = 0;
        for (size_t si = range.first; si < range.second; ++si) {
            auto &seq = seqs_[si];
            tokens_scanned += static_cast<uint64_t>(seq.size());
            sequences_scanned += 1;
            if (seq.size() < 2) {
                continue;
            }
            buffer.clear();
            bool changed = false;
            for (size_t i = 0; i < seq.size(); ) {
                if (i + 1 < seq.size() && seq[i] == best_left && seq[i + 1] == best_right) {
                    if (!changed) {
                        buffer.reserve(seq.size());
                        buffer.insert(buffer.end(), seq.begin(), seq.begin() + static_cast<long>(i));
                        changed = true;
                    }
                    buffer.push_back(new_id);
                    i += 2;
                } else {
                    if (changed) {
                        buffer.push_back(seq[i]);
                    }
                    i += 1;
                }
            }
            if (changed) {
                sequences_changed += 1;
                seq.swap(buffer);
            }
        }
        worker_pair_visits_[tid] = 0;
        worker_apply_tokens_[tid] = tokens_scanned;
        worker_apply_sequences_[tid] = sequences_scanned;
        worker_apply_changed_[tid] = sequences_changed;
    }

    void clear_counts() {
        for (uint32_t key : global_touched_) {
            global_counts_[key] = 0;
        }
        global_touched_.clear();

        for (size_t tid = 0; tid < num_threads_; ++tid) {
            auto &counts = local_counts_[tid];
            auto &touched = local_touched_[tid];
            for (uint32_t key : touched) {
                counts[key] = 0;
            }
            touched.clear();
        }
    }

    std::vector<std::vector<int32_t>> seqs_;
    size_t pair_dim_;
    size_t pair_space_;
    size_t num_threads_;
    std::vector<std::pair<size_t, size_t>> ranges_;
    std::vector<std::vector<uint32_t>> local_counts_;
    std::vector<std::vector<uint32_t>> local_touched_;
    std::vector<std::vector<int32_t>> apply_buffers_;
    std::vector<PairCount> global_counts_;
    std::vector<uint32_t> global_touched_;
    std::vector<int64_t> worker_elapsed_ns_ = std::vector<int64_t>(num_threads_, 0);
    std::vector<int64_t> worker_start_delay_ns_ = std::vector<int64_t>(num_threads_, 0);
    std::vector<uint64_t> worker_pair_visits_ = std::vector<uint64_t>(num_threads_, 0);
    std::vector<uint64_t> worker_apply_tokens_ = std::vector<uint64_t>(num_threads_, 0);
    std::vector<uint64_t> worker_apply_sequences_ = std::vector<uint64_t>(num_threads_, 0);
    std::vector<uint64_t> worker_apply_changed_ = std::vector<uint64_t>(num_threads_, 0);
    std::vector<std::thread> workers_;
    std::mutex mutex_;
    std::condition_variable cv_start_;
    std::condition_variable cv_done_;
    bool stop_ = false;
    Task task_ = Task::Count;
    size_t work_generation_ = 0;
    size_t completed_threads_ = 0;
    int32_t best_left_ = 0;
    int32_t best_right_ = 0;
    int32_t new_id_ = 0;
    Clock::time_point dispatch_time_ = Clock::now();
};

class DirectPairFrequencySampleParallelTrainer {
public:
    using LocalRightFrequencyRow = std::unordered_map<int32_t, uint32_t>;
    using GlobalRightFrequencyRow = std::unordered_map<int32_t, PairCount>;
    using LocalPairFrequencyRows = std::unordered_map<int32_t, LocalRightFrequencyRow>;
    using GlobalPairFrequencyRows = std::unordered_map<int32_t, GlobalRightFrequencyRow>;

    DirectPairFrequencySampleParallelTrainer(
        std::vector<std::vector<int32_t>> seqs,
        size_t pair_table_size,
        size_t num_threads
    )
        : seqs_(std::move(seqs)),
          pair_table_size_(pair_table_size),
          num_threads_(std::max<size_t>(1, std::min(num_threads, std::max<size_t>(1, seqs_.size())))),
          ranges_(make_ranges()),
          local_pair_frequencies_(num_threads_),
          local_unique_pairs_(num_threads_, 0),
          global_pair_frequencies_(),
          apply_buffers_(num_threads_)
    {
        if (pair_table_size_ == 0) {
            throw std::runtime_error("direct pair-frequency BPE requires a non-empty pair table");
        }
        for (size_t tid = 0; tid < num_threads_; ++tid) {
            local_pair_frequencies_[tid].reserve(65536);
        }
        global_pair_frequencies_.reserve(65536);
        workers_.reserve(num_threads_);
        for (size_t tid = 0; tid < num_threads_; ++tid) {
            workers_.emplace_back([this, tid]() { worker_loop(tid); });
        }
    }

    ~DirectPairFrequencySampleParallelTrainer() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_ = true;
            work_generation_ += 1;
        }
        cv_start_.notify_all();
        for (auto &worker : workers_) {
            if (worker.joinable()) {
                worker.join();
            }
        }
    }

    DirectPairFrequencySampleParallelTrainer(const DirectPairFrequencySampleParallelTrainer &) = delete;
    DirectPairFrequencySampleParallelTrainer &operator=(const DirectPairFrequencySampleParallelTrainer &) = delete;

    void count_best_pair(
        PairCount min_frequency,
        PairCount &best_freq,
        int32_t &best_left,
        int32_t &best_right,
        TimingProfile *profile = nullptr
    ) {
        const auto count_start = Clock::now();
        run_task(Task::Count, 0, 0, 0);
        if (profile != nullptr && profile->enabled) {
            profile->count_rounds += 1;
            const int64_t wall_ns = elapsed_ns(count_start);
            profile->count_wall_ns += wall_ns;
            uint64_t pairs = 0;
            uint64_t local_unique_pairs = 0;
            int64_t max_worker_ns = 0;
            int64_t max_start_delay_ns = 0;
            int64_t sum_start_delay_ns = 0;
            int64_t max_finish_ns = 0;
            for (size_t tid = 0; tid < num_threads_; ++tid) {
                pairs += worker_pair_visits_[tid];
                local_unique_pairs += static_cast<uint64_t>(local_unique_pairs_[tid]);
                profile->count_worker_sum_ns += worker_elapsed_ns_[tid];
                max_worker_ns = std::max(max_worker_ns, worker_elapsed_ns_[tid]);
                const int64_t start_delay_ns = worker_start_delay_ns_[tid];
                sum_start_delay_ns += start_delay_ns;
                max_start_delay_ns = std::max(max_start_delay_ns, start_delay_ns);
                max_finish_ns = std::max(max_finish_ns, start_delay_ns + worker_elapsed_ns_[tid]);
            }
            profile->count_pairs_visited += pairs;
            profile->local_touched_total += local_unique_pairs;
            profile->count_worker_max_ns += max_worker_ns;
            profile->count_start_delay_sum_ns += sum_start_delay_ns;
            profile->count_start_delay_max_ns += max_start_delay_ns;
            profile->count_active_span_max_ns += max_finish_ns;
            profile->count_tail_after_active_ns += wall_ns > max_finish_ns ? wall_ns - max_finish_ns : 0;
        }

        const auto reduce_start = Clock::now();
        global_pair_frequencies_.clear();
        uint64_t global_unique_pairs = 0;
        for (size_t tid = 0; tid < num_threads_; ++tid) {
            const auto &local_rows = local_pair_frequencies_[tid];
            for (const auto &left_item : local_rows) {
                const int32_t left = left_item.first;
                const auto &local_row = left_item.second;
                auto &global_row = global_pair_frequencies_[left];
                for (const auto &right_item : local_row) {
                    const int32_t right = right_item.first;
                    const uint32_t local_freq = right_item.second;
                    auto it = global_row.find(right);
                    if (it == global_row.end()) {
                        global_row.emplace(right, static_cast<PairCount>(local_freq));
                        global_unique_pairs += 1;
                    } else {
                        it->second += static_cast<PairCount>(local_freq);
                    }
                }
            }
        }
        if (profile != nullptr && profile->enabled) {
            profile->reduce_ns += elapsed_ns(reduce_start);
            profile->global_unique_total += global_unique_pairs;
        }

        const auto select_start = Clock::now();
        best_freq = -1;
        best_left = 0;
        best_right = 0;
        for (const auto &left_item : global_pair_frequencies_) {
            const int32_t left = left_item.first;
            for (const auto &right_item : left_item.second) {
                const int32_t right = right_item.first;
                const PairCount freq = right_item.second;
                if (freq < min_frequency) continue;
                if (pair_is_better(freq, left, right, best_freq, best_left, best_right)) {
                    best_freq = freq;
                    best_left = left;
                    best_right = right;
                }
            }
        }
        if (profile != nullptr && profile->enabled) {
            profile->select_ns += elapsed_ns(select_start);
        }

        const auto clear_start = Clock::now();
        global_pair_frequencies_.clear();
        if (profile != nullptr && profile->enabled) {
            profile->clear_ns += elapsed_ns(clear_start);
        }
    }

    void apply_merge(int32_t best_left, int32_t best_right, int32_t new_id, TimingProfile *profile = nullptr) {
        const auto apply_start = Clock::now();
        run_task(Task::Apply, best_left, best_right, new_id);
        if (profile != nullptr && profile->enabled) {
            profile->apply_rounds += 1;
            const int64_t wall_ns = elapsed_ns(apply_start);
            profile->apply_wall_ns += wall_ns;
            int64_t max_worker_ns = 0;
            int64_t max_start_delay_ns = 0;
            int64_t sum_start_delay_ns = 0;
            int64_t max_finish_ns = 0;
            for (size_t tid = 0; tid < num_threads_; ++tid) {
                profile->apply_worker_sum_ns += worker_elapsed_ns_[tid];
                max_worker_ns = std::max(max_worker_ns, worker_elapsed_ns_[tid]);
                const int64_t start_delay_ns = worker_start_delay_ns_[tid];
                sum_start_delay_ns += start_delay_ns;
                max_start_delay_ns = std::max(max_start_delay_ns, start_delay_ns);
                max_finish_ns = std::max(max_finish_ns, start_delay_ns + worker_elapsed_ns_[tid]);
                profile->apply_tokens_scanned += worker_apply_tokens_[tid];
                profile->apply_sequences_scanned += worker_apply_sequences_[tid];
                profile->apply_sequences_changed += worker_apply_changed_[tid];
            }
            profile->apply_worker_max_ns += max_worker_ns;
            profile->apply_start_delay_sum_ns += sum_start_delay_ns;
            profile->apply_start_delay_max_ns += max_start_delay_ns;
            profile->apply_active_span_max_ns += max_finish_ns;
            profile->apply_tail_after_active_ns += wall_ns > max_finish_ns ? wall_ns - max_finish_ns : 0;
        }
    }

    void fill_static_profile(TimingProfile &profile) const {
        profile.num_threads = static_cast<int32_t>(num_threads_);
        profile.input_sequences = static_cast<uint64_t>(seqs_.size());
        uint64_t min_tokens = std::numeric_limits<uint64_t>::max();
        uint64_t max_tokens = 0;
        uint64_t nonempty = 0;
        uint64_t total_tokens = 0;
        for (size_t tid = 0; tid < num_threads_; ++tid) {
            uint64_t range_tokens = 0;
            const auto range = ranges_[tid];
            for (size_t si = range.first; si < range.second; ++si) {
                range_tokens += static_cast<uint64_t>(seqs_[si].size());
            }
            if (range.second > range.first) {
                nonempty += 1;
            }
            min_tokens = std::min(min_tokens, range_tokens);
            max_tokens = std::max(max_tokens, range_tokens);
            total_tokens += range_tokens;
        }
        profile.input_tokens = total_tokens;
        profile.nonempty_threads = nonempty;
        profile.range_tokens_min = min_tokens == std::numeric_limits<uint64_t>::max() ? 0 : min_tokens;
        profile.range_tokens_max = max_tokens;
        profile.local_count_workspace_mb = static_cast<uint64_t>(
            (static_cast<long double>(num_threads_) * 65536.0L * sizeof(std::pair<const int32_t, uint32_t>))
            / (1024.0L * 1024.0L)
        );
        profile.global_count_workspace_mb = static_cast<uint64_t>(
            (65536.0L * sizeof(std::pair<const int32_t, PairCount>))
            / (1024.0L * 1024.0L)
        );
    }

    std::vector<std::array<int64_t, 3>> count_pair_frequencies() {
        run_task(Task::Count, 0, 0, 0);

        global_pair_frequencies_.clear();
        for (size_t tid = 0; tid < num_threads_; ++tid) {
            const auto &local_rows = local_pair_frequencies_[tid];
            for (const auto &left_item : local_rows) {
                const int32_t left = left_item.first;
                const auto &local_row = left_item.second;
                auto &global_row = global_pair_frequencies_[left];
                for (const auto &right_item : local_row) {
                    const int32_t right = right_item.first;
                    const uint32_t local_freq = right_item.second;
                    auto it = global_row.find(right);
                    if (it == global_row.end()) {
                        global_row.emplace(right, static_cast<PairCount>(local_freq));
                    } else {
                        it->second += static_cast<PairCount>(local_freq);
                    }
                }
            }
        }

        std::vector<std::array<int64_t, 3>> counts;
        for (const auto &left_item : global_pair_frequencies_) {
            const int32_t left = left_item.first;
            for (const auto &right_item : left_item.second) {
                const int32_t right = right_item.first;
                const PairCount freq = right_item.second;
                if (freq > 0) {
                    counts.push_back({
                        static_cast<int64_t>(left),
                        static_cast<int64_t>(right),
                        static_cast<int64_t>(freq),
                    });
                }
            }
        }
        global_pair_frequencies_.clear();
        return counts;
    }

    std::array<uint64_t, 4> apply_merge_and_get_stats(int32_t best_left, int32_t best_right, int32_t new_id) {
        run_task(Task::Apply, best_left, best_right, new_id);
        uint64_t tokens_scanned = 0;
        uint64_t sequences_scanned = 0;
        uint64_t sequences_changed = 0;
        for (size_t tid = 0; tid < num_threads_; ++tid) {
            tokens_scanned += worker_apply_tokens_[tid];
            sequences_scanned += worker_apply_sequences_[tid];
            sequences_changed += worker_apply_changed_[tid];
        }
        return {tokens_scanned, sequences_scanned, sequences_changed, token_count()};
    }

    uint64_t token_count() const {
        uint64_t total = 0;
        for (const auto &seq : seqs_) {
            total += static_cast<uint64_t>(seq.size());
        }
        return total;
    }

    uint64_t sequence_count() const {
        return static_cast<uint64_t>(seqs_.size());
    }

private:
    enum class Task {
        Count,
        Apply,
    };

    std::vector<std::pair<size_t, size_t>> make_ranges() const {
        std::vector<std::pair<size_t, size_t>> ranges;
        ranges.reserve(num_threads_);
        const size_t nseq = seqs_.size();
        if (nseq == 0) {
            ranges.assign(num_threads_, {0, 0});
            return ranges;
        }

        size_t total_tokens = 0;
        for (const auto &seq : seqs_) {
            total_tokens += seq.size();
        }

        size_t start = 0;
        size_t consumed_tokens = 0;
        for (size_t tid = 0; tid < num_threads_; ++tid) {
            if (start >= nseq) {
                ranges.push_back({nseq, nseq});
                continue;
            }
            if (tid + 1 == num_threads_) {
                ranges.push_back({start, nseq});
                break;
            }
            const size_t remaining_threads = num_threads_ - tid;
            const size_t remaining_tokens = total_tokens > consumed_tokens
                ? total_tokens - consumed_tokens
                : 0;
            const size_t target = std::max<size_t>(1, (remaining_tokens + remaining_threads - 1) / remaining_threads);
            size_t end = start;
            size_t cur_tokens = 0;
            while (end < nseq && (nseq - end) > (remaining_threads - 1)) {
                if (end > start && cur_tokens >= target) {
                    break;
                }
                cur_tokens += seqs_[end].size();
                ++end;
            }
            ranges.push_back({start, end});
            start = end;
            consumed_tokens += cur_tokens;
        }
        while (ranges.size() < num_threads_) {
            ranges.push_back({nseq, nseq});
        }
        return ranges;
    }

    void run_task(Task task, int32_t best_left, int32_t best_right, int32_t new_id) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            task_ = task;
            best_left_ = best_left;
            best_right_ = best_right;
            new_id_ = new_id;
            completed_threads_ = 0;
            dispatch_time_ = Clock::now();
            work_generation_ += 1;
        }
        cv_start_.notify_all();
        {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_done_.wait(lock, [&]() {
                return completed_threads_ >= num_threads_;
            });
        }
    }

    void worker_loop(size_t tid) {
        size_t seen_generation = 0;
        while (true) {
            Task task = Task::Count;
            int32_t best_left = 0;
            int32_t best_right = 0;
            int32_t new_id = 0;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                cv_start_.wait(lock, [&]() {
                    return stop_ || work_generation_ != seen_generation;
                });
                if (stop_) {
                    return;
                }
                seen_generation = work_generation_;
                task = task_;
                best_left = best_left_;
                best_right = best_right_;
                new_id = new_id_;
            }

            const auto task_start = Clock::now();
            worker_start_delay_ns_[tid] = std::chrono::duration_cast<std::chrono::nanoseconds>(
                task_start - dispatch_time_
            ).count();
            if (task == Task::Count) {
                count_range(tid);
            } else {
                apply_range(tid, best_left, best_right, new_id);
            }
            worker_elapsed_ns_[tid] = elapsed_ns(task_start);

            {
                std::lock_guard<std::mutex> lock(mutex_);
                completed_threads_ += 1;
                if (completed_threads_ >= num_threads_) {
                    cv_done_.notify_one();
                }
            }
        }
    }

    void count_range(size_t tid) {
        auto &rows = local_pair_frequencies_[tid];
        rows.clear();
        const auto range = ranges_[tid];
        uint64_t pair_visits = 0;
        uint64_t unique_pairs = 0;
        for (size_t si = range.first; si < range.second; ++si) {
            const auto &seq = seqs_[si];
            for (size_t i = 0; i + 1 < seq.size(); ++i) {
                const int32_t left = seq[i];
                const int32_t right = seq[i + 1];
                auto &row = rows[left];
                auto it = row.find(right);
                if (it == row.end()) {
                    row.emplace(right, 1);
                    unique_pairs += 1;
                } else {
                    it->second += 1;
                    if (it->second == 0) {
                        throw std::runtime_error("direct pair-frequency local counter overflow");
                    }
                }
                pair_visits += 1;
            }
        }
        worker_pair_visits_[tid] = pair_visits;
        local_unique_pairs_[tid] = unique_pairs;
        worker_apply_tokens_[tid] = 0;
        worker_apply_sequences_[tid] = 0;
        worker_apply_changed_[tid] = 0;
    }

    void apply_range(size_t tid, int32_t best_left, int32_t best_right, int32_t new_id) {
        auto &buffer = apply_buffers_[tid];
        const auto range = ranges_[tid];
        uint64_t tokens_scanned = 0;
        uint64_t sequences_scanned = 0;
        uint64_t sequences_changed = 0;
        for (size_t si = range.first; si < range.second; ++si) {
            auto &seq = seqs_[si];
            tokens_scanned += static_cast<uint64_t>(seq.size());
            sequences_scanned += 1;
            if (seq.size() < 2) {
                continue;
            }
            buffer.clear();
            bool changed = false;
            for (size_t i = 0; i < seq.size(); ) {
                if (i + 1 < seq.size() && seq[i] == best_left && seq[i + 1] == best_right) {
                    if (!changed) {
                        buffer.reserve(seq.size());
                        buffer.insert(buffer.end(), seq.begin(), seq.begin() + static_cast<long>(i));
                        changed = true;
                    }
                    buffer.push_back(new_id);
                    i += 2;
                } else {
                    if (changed) {
                        buffer.push_back(seq[i]);
                    }
                    i += 1;
                }
            }
            if (changed) {
                sequences_changed += 1;
                seq.swap(buffer);
            }
        }
        worker_pair_visits_[tid] = 0;
        worker_apply_tokens_[tid] = tokens_scanned;
        worker_apply_sequences_[tid] = sequences_scanned;
        worker_apply_changed_[tid] = sequences_changed;
    }

    void clear_counts() {
        global_pair_frequencies_.clear();
    }

    std::vector<std::vector<int32_t>> seqs_;
    size_t pair_table_size_;
    size_t num_threads_;
    std::vector<std::pair<size_t, size_t>> ranges_;
    std::vector<LocalPairFrequencyRows> local_pair_frequencies_;
    std::vector<uint64_t> local_unique_pairs_;
    GlobalPairFrequencyRows global_pair_frequencies_;
    std::vector<std::vector<int32_t>> apply_buffers_;
    std::vector<int64_t> worker_elapsed_ns_ = std::vector<int64_t>(num_threads_, 0);
    std::vector<int64_t> worker_start_delay_ns_ = std::vector<int64_t>(num_threads_, 0);
    std::vector<uint64_t> worker_pair_visits_ = std::vector<uint64_t>(num_threads_, 0);
    std::vector<uint64_t> worker_apply_tokens_ = std::vector<uint64_t>(num_threads_, 0);
    std::vector<uint64_t> worker_apply_sequences_ = std::vector<uint64_t>(num_threads_, 0);
    std::vector<uint64_t> worker_apply_changed_ = std::vector<uint64_t>(num_threads_, 0);
    std::vector<std::thread> workers_;
    std::mutex mutex_;
    std::condition_variable cv_start_;
    std::condition_variable cv_done_;
    bool stop_ = false;
    Task task_ = Task::Count;
    size_t work_generation_ = 0;
    size_t completed_threads_ = 0;
    int32_t best_left_ = 0;
    int32_t best_right_ = 0;
    int32_t new_id_ = 0;
    Clock::time_point dispatch_time_ = Clock::now();
};

template <typename CounterT>
TrainResult train_minbpe_dense_parallel_impl_t(
    const std::vector<std::vector<int32_t>> &seqs,
    int32_t num_merges,
    int32_t min_frequency,
    int32_t requested_threads,
    bool parallel_apply,
    bool reuse_serial_apply
) {
    std::unordered_set<int32_t> base_vocab;
    base_vocab.reserve(1024);
    int32_t max_base_id = -1;
    for (const auto &s : seqs) {
        for (int32_t t : s) {
            if (t < 0) {
                throw std::runtime_error("dense BPE only supports non-negative token ids");
            }
            base_vocab.insert(t);
            if (t > max_base_id) max_base_id = t;
        }
    }
    const int32_t separator = max_base_id + 1;
    int32_t next_id = separator + 1;

    std::vector<int32_t> ids;
    size_t total_len = 0;
    for (const auto &s : seqs) total_len += s.size();
    if (!seqs.empty()) total_len += (seqs.size() - 1);
    ids.reserve(total_len);
    for (size_t i = 0; i < seqs.size(); ++i) {
        if (i > 0) ids.push_back(separator);
        const auto &s = seqs[i];
        ids.insert(ids.end(), s.begin(), s.end());
    }

    unsigned int hardware_threads = std::thread::hardware_concurrency();
    if (hardware_threads == 0) hardware_threads = 1;
    size_t num_threads = requested_threads > 0
        ? static_cast<size_t>(requested_threads)
        : static_cast<size_t>(hardware_threads);
    num_threads = std::max<size_t>(1, num_threads);

    const size_t pair_dim = static_cast<size_t>(separator) + static_cast<size_t>(num_merges) + 1;
    CounterT counter(pair_dim, num_threads);

    std::vector<std::array<int32_t, 3>> merges;
    merges.reserve(static_cast<size_t>(num_merges));
    std::vector<int32_t> merge_buffer;

    for (int32_t step = 0; step < num_merges; ++step) {
        PairCount best_freq = -1;
        int32_t best_left = 0;
        int32_t best_right = 0;
        counter.count(
            ids,
            separator,
            static_cast<PairCount>(min_frequency),
            best_freq,
            best_left,
            best_right
        );
        if (best_freq < min_frequency) {
            break;
        }

        const int32_t new_id = next_id++;
        if (parallel_apply) {
            if (best_left == best_right) {
                apply_merge_serial_reuse(ids, best_left, best_right, new_id, merge_buffer);
                ids.swap(merge_buffer);
            } else {
                std::vector<int32_t> out = apply_merge_parallel_distinct(
                    ids,
                    best_left,
                    best_right,
                    new_id,
                    num_threads
                );
                ids.swap(out);
            }
        } else if (reuse_serial_apply) {
            apply_merge_serial_reuse(ids, best_left, best_right, new_id, merge_buffer);
            ids.swap(merge_buffer);
        } else {
            std::vector<int32_t> out;
            out.reserve(ids.size());
            for (size_t i = 0; i < ids.size(); ) {
                if (i + 1 < ids.size() && ids[i] == best_left && ids[i + 1] == best_right) {
                    out.push_back(new_id);
                    i += 2;
                } else {
                    out.push_back(ids[i]);
                    i += 1;
                }
            }
            ids.swap(out);
        }
        merges.push_back({best_left, best_right, new_id});
    }

    TrainResult result;
    result.merges = std::move(merges);
    result.base_vocab_size = static_cast<int32_t>(base_vocab.size());
    result.final_vocab_size = result.base_vocab_size + static_cast<int32_t>(result.merges.size());
    result.separator_token = separator;
    return result;
}

TrainResult train_minbpe_dense_parallel_impl(
    const std::vector<std::vector<int32_t>> &seqs,
    int32_t num_merges,
    int32_t min_frequency,
    int32_t requested_threads,
    bool parallel_apply,
    bool reuse_serial_apply
) {
    return train_minbpe_dense_parallel_impl_t<DensePairCounter>(
        seqs,
        num_merges,
        min_frequency,
        requested_threads,
        parallel_apply,
        reuse_serial_apply
    );
}

TrainResult train_minbpe_dense_parallel_pool_impl(
    const std::vector<std::vector<int32_t>> &seqs,
    int32_t num_merges,
    int32_t min_frequency,
    int32_t requested_threads
) {
    return train_minbpe_dense_parallel_impl_t<DensePairCounterPool>(
        seqs,
        num_merges,
        min_frequency,
        requested_threads,
        false,
        false
    );
}

TrainResult train_minbpe_dense_sample_parallel_impl(
    std::vector<std::vector<int32_t>> seqs,
    int32_t num_merges,
    int32_t min_frequency,
    int32_t requested_threads
) {
    const auto base_vocab_start = Clock::now();
    std::unordered_set<int32_t> base_vocab;
    base_vocab.reserve(1024);
    int32_t max_base_id = -1;
    for (const auto &s : seqs) {
        for (int32_t t : s) {
            if (t < 0) {
                throw std::runtime_error("dense sample-parallel BPE only supports non-negative token ids");
            }
            base_vocab.insert(t);
            if (t > max_base_id) max_base_id = t;
        }
    }

    // Keep the historical new-id offset for codebook compatibility with the existing C++ backends.
    // The reserved id is not inserted into sequences and is not used for pair counting or apply.
    const int32_t reserved_separator_token = max_base_id + 1;
    int32_t next_id = reserved_separator_token + 1;
    const int64_t base_vocab_scan_ns = elapsed_ns(base_vocab_start);

    unsigned int hardware_threads = std::thread::hardware_concurrency();
    if (hardware_threads == 0) hardware_threads = 1;
    size_t num_threads = requested_threads > 0
        ? static_cast<size_t>(requested_threads)
        : static_cast<size_t>(hardware_threads);
    num_threads = std::max<size_t>(1, num_threads);

    const size_t pair_dim = static_cast<size_t>(reserved_separator_token) + static_cast<size_t>(num_merges) + 1;
    const auto trainer_setup_start = Clock::now();
    DenseSampleParallelTrainer trainer(std::move(seqs), pair_dim, num_threads);
    const int64_t trainer_setup_ns = elapsed_ns(trainer_setup_start);
    TimingProfile profile;
    profile.enabled = true;
    profile.base_vocab_scan_ns = base_vocab_scan_ns;
    profile.trainer_setup_ns = trainer_setup_ns;
    trainer.fill_static_profile(profile);

    std::vector<std::array<int32_t, 3>> merges;
    merges.reserve(static_cast<size_t>(num_merges));

    const auto merge_loop_start = Clock::now();
    for (int32_t step = 0; step < num_merges; ++step) {
        PairCount best_freq = -1;
        int32_t best_left = 0;
        int32_t best_right = 0;
        trainer.count_best_pair(
            static_cast<PairCount>(min_frequency),
            best_freq,
            best_left,
            best_right,
            &profile
        );
        if (best_freq < min_frequency) {
            break;
        }

        const int32_t new_id = next_id++;
        trainer.apply_merge(best_left, best_right, new_id, &profile);
        merges.push_back({best_left, best_right, new_id});
        profile.completed_merges = static_cast<int32_t>(merges.size());
    }
    profile.merge_loop_ns = elapsed_ns(merge_loop_start);

    TrainResult result;
    result.merges = std::move(merges);
    result.base_vocab_size = static_cast<int32_t>(base_vocab.size());
    result.final_vocab_size = result.base_vocab_size + static_cast<int32_t>(result.merges.size());
    result.separator_token = reserved_separator_token;
    result.profile = profile;
    return result;
}

TrainResult train_minbpe_direct_pair_sample_parallel_impl(
    std::vector<std::vector<int32_t>> seqs,
    int32_t num_merges,
    int32_t min_frequency,
    int32_t requested_threads
) {
    const auto base_vocab_start = Clock::now();
    std::unordered_set<int32_t> base_vocab;
    base_vocab.reserve(1024);
    int32_t max_base_id = -1;
    for (const auto &s : seqs) {
        for (int32_t t : s) {
            if (t < 0) {
                throw std::runtime_error("direct pair-frequency sample-parallel BPE only supports non-negative token ids");
            }
            base_vocab.insert(t);
            if (t > max_base_id) max_base_id = t;
        }
    }
    const int32_t reserved_separator_token = max_base_id + 1;
    int32_t next_id = reserved_separator_token + 1;
    const int64_t base_vocab_scan_ns = elapsed_ns(base_vocab_start);

    unsigned int hardware_threads = std::thread::hardware_concurrency();
    if (hardware_threads == 0) hardware_threads = 1;
    size_t num_threads = requested_threads > 0
        ? static_cast<size_t>(requested_threads)
        : static_cast<size_t>(hardware_threads);
    num_threads = std::max<size_t>(1, std::min(num_threads, std::max<size_t>(1, seqs.size())));

    const auto setup_start = Clock::now();
    const size_t pair_table_size = static_cast<size_t>(reserved_separator_token)
        + static_cast<size_t>(std::max<int32_t>(0, num_merges)) + 1;
    DirectPairFrequencySampleParallelTrainer trainer(std::move(seqs), pair_table_size, num_threads);
    const int64_t setup_ns = elapsed_ns(setup_start);

    TimingProfile profile;
    profile.enabled = true;
    profile.base_vocab_scan_ns = base_vocab_scan_ns;
    profile.trainer_setup_ns = setup_ns;
    trainer.fill_static_profile(profile);

    std::vector<std::array<int32_t, 3>> merges;
    merges.reserve(static_cast<size_t>(num_merges));
    const auto loop_start = Clock::now();

    for (int32_t step = 0; step < num_merges; ++step) {
        PairCount best_freq = -1;
        int32_t best_left = 0;
        int32_t best_right = 0;
        trainer.count_best_pair(
            static_cast<PairCount>(min_frequency),
            best_freq,
            best_left,
            best_right,
            &profile
        );
        if (best_freq < min_frequency) {
            break;
        }

        const int32_t new_id = next_id++;
        trainer.apply_merge(best_left, best_right, new_id, &profile);
        merges.push_back({best_left, best_right, new_id});
        profile.completed_merges = static_cast<int32_t>(merges.size());
    }
    profile.merge_loop_ns = elapsed_ns(loop_start);

    TrainResult result;
    result.merges = std::move(merges);
    result.base_vocab_size = static_cast<int32_t>(base_vocab.size());
    result.final_vocab_size = result.base_vocab_size + static_cast<int32_t>(result.merges.size());
    result.separator_token = reserved_separator_token;
    result.profile = profile;
    return result;
}

py::dict train_result_to_dict(const TrainResult &result) {
    py::list py_merges;
    for (const auto &merge : result.merges) {
        py_merges.append(py::make_tuple(merge[0], merge[1], merge[2]));
    }
    py::dict ret;
    ret[py::str("merge_rules")] = py_merges;
    ret[py::str("num_merges_performed")] = static_cast<int32_t>(result.merges.size());
    ret[py::str("final_vocab_size")] = result.final_vocab_size;
    ret[py::str("base_vocab_size")] = result.base_vocab_size;
    ret[py::str("separator_token")] = result.separator_token;
    if (result.profile.enabled) {
        py::dict profile;
        profile[py::str("num_threads")] = result.profile.num_threads;
        profile[py::str("completed_merges")] = result.profile.completed_merges;
        profile[py::str("input_sequences")] = result.profile.input_sequences;
        profile[py::str("input_tokens")] = result.profile.input_tokens;
        profile[py::str("nonempty_threads")] = result.profile.nonempty_threads;
        profile[py::str("range_tokens_min")] = result.profile.range_tokens_min;
        profile[py::str("range_tokens_max")] = result.profile.range_tokens_max;
        profile[py::str("local_count_workspace_mb")] = result.profile.local_count_workspace_mb;
        profile[py::str("global_count_workspace_mb")] = result.profile.global_count_workspace_mb;
        profile[py::str("base_vocab_scan_ns")] = result.profile.base_vocab_scan_ns;
        profile[py::str("trainer_setup_ns")] = result.profile.trainer_setup_ns;
        profile[py::str("merge_loop_ns")] = result.profile.merge_loop_ns;
        profile[py::str("count_rounds")] = result.profile.count_rounds;
        profile[py::str("apply_rounds")] = result.profile.apply_rounds;
        profile[py::str("count_pairs_visited")] = result.profile.count_pairs_visited;
        profile[py::str("local_touched_total")] = result.profile.local_touched_total;
        profile[py::str("global_unique_total")] = result.profile.global_unique_total;
        profile[py::str("apply_tokens_scanned")] = result.profile.apply_tokens_scanned;
        profile[py::str("apply_sequences_scanned")] = result.profile.apply_sequences_scanned;
        profile[py::str("apply_sequences_changed")] = result.profile.apply_sequences_changed;
        profile[py::str("count_wall_ns")] = result.profile.count_wall_ns;
        profile[py::str("count_start_delay_sum_ns")] = result.profile.count_start_delay_sum_ns;
        profile[py::str("count_start_delay_max_ns")] = result.profile.count_start_delay_max_ns;
        profile[py::str("count_active_span_max_ns")] = result.profile.count_active_span_max_ns;
        profile[py::str("count_tail_after_active_ns")] = result.profile.count_tail_after_active_ns;
        profile[py::str("count_worker_sum_ns")] = result.profile.count_worker_sum_ns;
        profile[py::str("count_worker_max_ns")] = result.profile.count_worker_max_ns;
        profile[py::str("reduce_ns")] = result.profile.reduce_ns;
        profile[py::str("select_ns")] = result.profile.select_ns;
        profile[py::str("clear_ns")] = result.profile.clear_ns;
        profile[py::str("apply_wall_ns")] = result.profile.apply_wall_ns;
        profile[py::str("apply_start_delay_sum_ns")] = result.profile.apply_start_delay_sum_ns;
        profile[py::str("apply_start_delay_max_ns")] = result.profile.apply_start_delay_max_ns;
        profile[py::str("apply_active_span_max_ns")] = result.profile.apply_active_span_max_ns;
        profile[py::str("apply_tail_after_active_ns")] = result.profile.apply_tail_after_active_ns;
        profile[py::str("apply_worker_sum_ns")] = result.profile.apply_worker_sum_ns;
        profile[py::str("apply_worker_max_ns")] = result.profile.apply_worker_max_ns;
        ret[py::str("profile")] = profile;
    }
    return ret;
}

}  // namespace

class MinBPEEncoder {
public:
    MinBPEEncoder(const std::vector<int32_t> &lefts,
                  const std::vector<int32_t> &rights,
                  const std::vector<int32_t> &ranks,
                  const std::vector<int32_t> &new_ids)
    {
        const size_t n = lefts.size();
        int32_t max_id = 0;
        for (size_t i = 0; i < n; ++i) {
            if (lefts[i] > max_id) max_id = lefts[i];
            if (rights[i] > max_id) max_id = rights[i];
        }
        // 每个 left 一个桶，右侧用 unordered_map（局部小表，查找更快）
        rank_tbl_.assign(static_cast<size_t>(max_id) + 1, {});
        newid_tbl_.assign(static_cast<size_t>(max_id) + 1, {});
        for (size_t i = 0; i < n; ++i) {
            auto L = static_cast<size_t>(lefts[i]);
            rank_tbl_[L][rights[i]] = ranks[i];
            newid_tbl_[L][rights[i]] = new_ids[i];
        }
    }

    std::vector<int32_t> encode(const std::vector<int32_t> &seq_in) const {
        return encode_with_limit(seq_in, INT32_C(0x7fffffff));
    }

    std::vector<int32_t> encode_with_limit(const std::vector<int32_t> &seq_in, int32_t rank_limit) const {
        if (seq_in.empty()) return {};
        std::vector<int32_t> seq = seq_in;            // 工作缓冲 A
        std::vector<int32_t> out; out.resize(seq.size()); // 工作缓冲 B（容量一次到位）

        while (true) {
            // 1) 找最小 rank 的 pair
            int32_t best_rank = INT32_C(0x7fffffff);
            int32_t best_l = -1, best_r = -1;
            const size_t n = seq.size();
            if (n < 2) break;
            for (size_t i = 0; i + 1 < n; ++i) {
                int32_t l = seq[i];
                int32_t r = seq[i + 1];
                if (l >= 0 && static_cast<size_t>(l) < rank_tbl_.size()) {
                    const auto &mp = rank_tbl_[static_cast<size_t>(l)];
                    auto it = mp.find(r);
                    if (it != mp.end()) {
                        int32_t rank = it->second;
                        if (rank >= rank_limit) continue;
                        if (rank < best_rank) {
                            best_rank = rank;
                            best_l = l;
                            best_r = r;
                        }
                    }
                }
            }
            if (best_rank == INT32_C(0x7fffffff)) {
                break;
            }

            // 2) 左到右执行该 pair 的合并
            size_t out_len = 0;
            for (size_t i = 0; i < n; ) {
                if (i + 1 < n && seq[i] == best_l && seq[i + 1] == best_r) {
                    const auto &mpn = newid_tbl_[static_cast<size_t>(best_l)];
                    auto itn = mpn.find(best_r);
                    int32_t new_id = itn->second;
                    out[out_len++] = new_id;
                    i += 2;
                } else {
                    out[out_len++] = seq[i];
                    i += 1;
                }
            }
            // 3) 复用缓冲并交换，避免重复分配/拷贝
            seq.resize(out_len);
            std::copy(out.begin(), out.begin() + static_cast<long>(out_len), seq.begin());
        }

        return seq;
    }

    py::tuple encode_with_limit_trace(const std::vector<int32_t> &seq_in, int32_t rank_limit) const {
        if (seq_in.empty()) return py::make_tuple(std::vector<int32_t>{}, std::vector<int32_t>{});
        std::vector<int32_t> seq = seq_in;
        std::vector<int32_t> out; out.resize(seq.size());
        std::vector<int32_t> ranks_applied;

        while (true) {
            int32_t best_rank = INT32_C(0x7fffffff);
            int32_t best_l = -1, best_r = -1;
            const size_t n = seq.size();
            if (n < 2) break;
            for (size_t i = 0; i + 1 < n; ++i) {
                int32_t l = seq[i];
                int32_t r = seq[i + 1];
                if (l >= 0 && static_cast<size_t>(l) < rank_tbl_.size()) {
                    const auto &mp = rank_tbl_[static_cast<size_t>(l)];
                    auto it = mp.find(r);
                    if (it != mp.end()) {
                        int32_t rank = it->second;
                        if (rank >= rank_limit) continue;
                        if (rank < best_rank) {
                            best_rank = rank;
                            best_l = l;
                            best_r = r;
                        }
                    }
                }
            }
            if (best_rank == INT32_C(0x7fffffff)) {
                break;
            }
            ranks_applied.push_back(best_rank);

            size_t out_len = 0;
            for (size_t i = 0; i < n; ) {
                if (i + 1 < n && seq[i] == best_l && seq[i + 1] == best_r) {
                    const auto &mpn = newid_tbl_[static_cast<size_t>(best_l)];
                    auto itn = mpn.find(best_r);
                    int32_t new_id = itn->second;
                    out[out_len++] = new_id;
                    i += 2;
                } else {
                    out[out_len++] = seq[i];
                    i += 1;
                }
            }
            seq.resize(out_len);
            std::copy(out.begin(), out.begin() + static_cast<long>(out_len), seq.begin());
        }
        return py::make_tuple(seq, ranks_applied);
    }

    std::vector<std::vector<int32_t>> batch_encode(const std::vector<std::vector<int32_t>> &seqs) const {
        std::vector<std::vector<int32_t>> results;
        results.reserve(seqs.size());
        for (const auto &s : seqs) {
            results.emplace_back(encode_with_limit(s, INT32_C(0x7fffffff)));
        }
        return results;
    }

    std::vector<std::vector<int32_t>> batch_encode_with_limit(const std::vector<std::vector<int32_t>> &seqs, int32_t rank_limit) const {
        std::vector<std::vector<int32_t>> results;
        results.reserve(seqs.size());
        for (const auto &s : seqs) {
            results.emplace_back(encode_with_limit(s, rank_limit));
        }
        return results;
    }

    // P2: ragged 批接口。输入为扁平 ids 与 offsets（长度 nseq+1，最后一个为 flat.size()）。
    // 返回 (flat_out, out_offsets)
    py::tuple encode_ragged(const std::vector<int32_t> &flat,
                            const std::vector<int32_t> &offsets) const {
        const size_t nseq = offsets.size() > 0 ? offsets.size() - 1 : 0;
        std::vector<int32_t> out_flat;
        out_flat.reserve(flat.size());
        std::vector<int32_t> out_offsets;
        out_offsets.reserve(offsets.size());
        out_offsets.push_back(0);

        for (size_t i = 0; i < nseq; ++i) {
            const size_t b = static_cast<size_t>(offsets[i]);
            const size_t e = static_cast<size_t>(offsets[i + 1]);
            std::vector<int32_t> seq;
            seq.assign(flat.begin() + static_cast<long>(b), flat.begin() + static_cast<long>(e));
            auto enc = encode(seq);
            out_flat.insert(out_flat.end(), enc.begin(), enc.end());
            out_offsets.push_back(static_cast<int32_t>(out_flat.size()));
        }
        return py::make_tuple(out_flat, out_offsets);
    }

private:
    // 左 token → (右 token → rank/newid)
    std::vector<std::unordered_map<int32_t, int32_t>> rank_tbl_;
    std::vector<std::unordered_map<int32_t, int32_t>> newid_tbl_;
};

PYBIND11_MODULE(_cpp_bpe, m) {
    py::class_<MinBPEEncoder>(m, "MinBPEEncoder")
        .def(py::init<const std::vector<int32_t>&,
                      const std::vector<int32_t>&,
                      const std::vector<int32_t>&,
                      const std::vector<int32_t>&>())
        .def("encode", &MinBPEEncoder::encode)
        .def("encode_with_limit", &MinBPEEncoder::encode_with_limit)
        .def("encode_with_limit_trace", &MinBPEEncoder::encode_with_limit_trace)
        .def("batch_encode", &MinBPEEncoder::batch_encode)
        .def("batch_encode_with_limit", &MinBPEEncoder::batch_encode_with_limit)
        .def("encode_ragged", &MinBPEEncoder::encode_ragged);

    // 训练：minBPE 逻辑（每轮全量统计 + 非重叠合并），返回 merge 规则与最终词表大小（不包含分隔符）。
    m.def("train_minbpe", [](const std::vector<std::vector<int32_t>> &seqs,
                             int32_t num_merges,
                             int32_t min_frequency) {
        TrainResult result = train_minbpe_impl(seqs, num_merges, min_frequency, false, 1);
        return train_result_to_dict(result);
    }, py::arg("seqs"), py::arg("num_merges"), py::arg("min_frequency"));

    m.def("train_minbpe_parallel", [](const std::vector<std::vector<int32_t>> &seqs,
                                      int32_t num_merges,
                                      int32_t min_frequency,
                                      int32_t num_threads) {
        TrainResult result;
        {
            py::gil_scoped_release release;
            result = train_minbpe_impl(seqs, num_merges, min_frequency, true, num_threads);
        }
        return train_result_to_dict(result);
    }, py::arg("seqs"), py::arg("num_merges"), py::arg("min_frequency"), py::arg("num_threads") = 0);

    m.def("train_minbpe_dense_parallel", [](const std::vector<std::vector<int32_t>> &seqs,
                                            int32_t num_merges,
                                            int32_t min_frequency,
                                            int32_t num_threads) {
        TrainResult result;
        {
            py::gil_scoped_release release;
            result = train_minbpe_dense_parallel_impl(seqs, num_merges, min_frequency, num_threads, false, false);
        }
        return train_result_to_dict(result);
    }, py::arg("seqs"), py::arg("num_merges"), py::arg("min_frequency"), py::arg("num_threads") = 0);

    m.def("train_minbpe_dense_parallel_apply", [](const std::vector<std::vector<int32_t>> &seqs,
                                                  int32_t num_merges,
                                                  int32_t min_frequency,
                                                  int32_t num_threads) {
        TrainResult result;
        {
            py::gil_scoped_release release;
            result = train_minbpe_dense_parallel_impl(seqs, num_merges, min_frequency, num_threads, true, false);
        }
        return train_result_to_dict(result);
    }, py::arg("seqs"), py::arg("num_merges"), py::arg("min_frequency"), py::arg("num_threads") = 0);

    m.def("train_minbpe_dense_parallel_reuse", [](const std::vector<std::vector<int32_t>> &seqs,
                                                  int32_t num_merges,
                                                  int32_t min_frequency,
                                                  int32_t num_threads) {
        TrainResult result;
        {
            py::gil_scoped_release release;
            result = train_minbpe_dense_parallel_impl(seqs, num_merges, min_frequency, num_threads, false, true);
        }
        return train_result_to_dict(result);
    }, py::arg("seqs"), py::arg("num_merges"), py::arg("min_frequency"), py::arg("num_threads") = 0);

    m.def("train_minbpe_dense_parallel_pool", [](const std::vector<std::vector<int32_t>> &seqs,
                                                 int32_t num_merges,
                                                 int32_t min_frequency,
                                                 int32_t num_threads) {
        TrainResult result;
        {
            py::gil_scoped_release release;
            result = train_minbpe_dense_parallel_pool_impl(seqs, num_merges, min_frequency, num_threads);
        }
        return train_result_to_dict(result);
    }, py::arg("seqs"), py::arg("num_merges"), py::arg("min_frequency"), py::arg("num_threads") = 0);

    m.def("train_minbpe_dense_sample_parallel", [](const std::vector<std::vector<int32_t>> &seqs,
                                                   int32_t num_merges,
                                                   int32_t min_frequency,
                                                   int32_t num_threads) {
        TrainResult result;
        {
            py::gil_scoped_release release;
            result = train_minbpe_dense_sample_parallel_impl(seqs, num_merges, min_frequency, num_threads);
        }
        return train_result_to_dict(result);
    }, py::arg("seqs"), py::arg("num_merges"), py::arg("min_frequency"), py::arg("num_threads") = 0);

    auto train_direct_pair_sample_parallel = [](const std::vector<std::vector<int32_t>> &seqs,
                                                int32_t num_merges,
                                                int32_t min_frequency,
                                                int32_t num_threads) {
        TrainResult result;
        {
            py::gil_scoped_release release;
            result = train_minbpe_direct_pair_sample_parallel_impl(seqs, num_merges, min_frequency, num_threads);
        }
        return train_result_to_dict(result);
    };
    m.def("train_minbpe_direct_pair_sample_parallel", train_direct_pair_sample_parallel,
          py::arg("seqs"), py::arg("num_merges"), py::arg("min_frequency"), py::arg("num_threads") = 0);
    m.def("train_minbpe_pairmap_sample_parallel", train_direct_pair_sample_parallel,
          py::arg("seqs"), py::arg("num_merges"), py::arg("min_frequency"), py::arg("num_threads") = 0);

    py::class_<DirectPairFrequencySampleParallelTrainer>(m, "DirectPairSampleShardTrainer")
        .def(py::init<std::vector<std::vector<int32_t>>, size_t, size_t>(),
             py::arg("seqs"), py::arg("pair_table_size"), py::arg("num_threads"))
        .def("count_pair_frequencies", [](DirectPairFrequencySampleParallelTrainer &trainer) {
            std::vector<std::array<int64_t, 3>> result;
            {
                py::gil_scoped_release release;
                result = trainer.count_pair_frequencies();
            }
            return result;
        })
        .def("apply_merge", [](DirectPairFrequencySampleParallelTrainer &trainer,
                               int32_t best_left,
                               int32_t best_right,
                               int32_t new_id) {
            std::array<uint64_t, 4> result;
            {
                py::gil_scoped_release release;
                result = trainer.apply_merge_and_get_stats(best_left, best_right, new_id);
            }
            return py::make_tuple(result[0], result[1], result[2], result[3]);
        }, py::arg("best_left"), py::arg("best_right"), py::arg("new_id"))
        .def("token_count", &DirectPairFrequencySampleParallelTrainer::token_count)
        .def("sequence_count", &DirectPairFrequencySampleParallelTrainer::sequence_count);
}
