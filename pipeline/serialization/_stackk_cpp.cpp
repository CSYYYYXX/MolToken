#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cctype>
#include <functional>
#include <map>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace {

using NodeKey = std::array<int64_t, 6>;
using EdgeKey = std::array<int64_t, 3>;
using TripletKey = std::array<int64_t, 15>;

struct NodeHash {
    size_t operator()(const NodeKey &key) const noexcept {
        size_t h = 1469598103934665603ULL;
        for (int64_t x : key) {
            h ^= static_cast<size_t>(x + 0x9e3779b97f4a7c15ULL);
            h *= 1099511628211ULL;
        }
        return h;
    }
};

struct EdgeHash {
    size_t operator()(const EdgeKey &key) const noexcept {
        size_t h = 1469598103934665603ULL;
        for (int64_t x : key) {
            h ^= static_cast<size_t>(x + 0x9e3779b97f4a7c15ULL);
            h *= 1099511628211ULL;
        }
        return h;
    }
};

struct TripletHash {
    size_t operator()(const TripletKey &key) const noexcept {
        size_t h = 1469598103934665603ULL;
        for (int64_t x : key) {
            h ^= static_cast<size_t>(x + 0x9e3779b97f4a7c15ULL);
            h *= 1099511628211ULL;
        }
        return h;
    }
};

struct UndirectedEdge {
    int edge_id = -1;
    int u = -1;
    int v = -1;
    EdgeKey feat{};

    int other(int node) const {
        if (node == u) return v;
        if (node == v) return u;
        throw std::runtime_error("node is not incident to edge");
    }
};

struct TransitionKey {
    int64_t triplet_count = 0;
    int64_t edge_count = 0;
    int64_t node_count = 0;
    std::string label;
    int neg_dst = 0;
};

struct RootKey {
    int64_t node_count = 0;
    int degree = 0;
    std::string label;
    int neg_node = 0;
};

bool transition_greater(const TransitionKey &a, const TransitionKey &b) {
    if (a.triplet_count != b.triplet_count) return a.triplet_count > b.triplet_count;
    if (a.edge_count != b.edge_count) return a.edge_count > b.edge_count;
    if (a.node_count != b.node_count) return a.node_count > b.node_count;
    if (a.label != b.label) return a.label > b.label;
    return a.neg_dst > b.neg_dst;
}

bool root_greater(const RootKey &a, const RootKey &b) {
    if (a.node_count != b.node_count) return a.node_count > b.node_count;
    if (a.degree != b.degree) return a.degree > b.degree;
    if (a.label != b.label) return a.label > b.label;
    return a.neg_node > b.neg_node;
}

NodeKey parse_node_key(py::handle obj) {
    NodeKey out{};
    py::sequence seq = py::reinterpret_borrow<py::sequence>(obj);
    if (py::len(seq) != 6) {
        throw std::runtime_error("node feature key must have length 6");
    }
    for (size_t i = 0; i < 6; ++i) out[i] = py::cast<int64_t>(seq[i]);
    return out;
}

EdgeKey parse_edge_key(py::handle obj) {
    EdgeKey out{};
    py::sequence seq = py::reinterpret_borrow<py::sequence>(obj);
    if (py::len(seq) != 3) {
        throw std::runtime_error("edge feature key must have length 3");
    }
    for (size_t i = 0; i < 3; ++i) out[i] = py::cast<int64_t>(seq[i]);
    return out;
}

TripletKey make_triplet_key(const NodeKey &src, const EdgeKey &edge, const NodeKey &dst) {
    TripletKey key{};
    for (size_t i = 0; i < 6; ++i) key[i] = src[i];
    for (size_t i = 0; i < 3; ++i) key[6 + i] = edge[i];
    for (size_t i = 0; i < 6; ++i) key[9 + i] = dst[i];
    return key;
}

std::string atom_symbol(int64_t atomic_num) {
    switch (atomic_num) {
        case 5: return "B";
        case 6: return "C";
        case 7: return "N";
        case 8: return "O";
        case 9: return "F";
        case 14: return "Si";
        case 15: return "P";
        case 16: return "S";
        case 17: return "Cl";
        case 35: return "Br";
        case 53: return "I";
        default: return "X" + std::to_string(atomic_num);
    }
}

bool can_aromatic_lower(const std::string &symbol) {
    return symbol == "B" || symbol == "C" || symbol == "N" || symbol == "O" || symbol == "P" || symbol == "S";
}

std::string compact_node_label(const NodeKey &feat) {
    std::string symbol = atom_symbol(feat[0]);
    if (feat[2] != 0 && can_aromatic_lower(symbol)) {
        std::transform(symbol.begin(), symbol.end(), symbol.begin(), [](unsigned char c) {
            return static_cast<char>(std::tolower(c));
        });
    }
    const int64_t charge = feat[1];
    if (charge > 0) {
        symbol += (charge == 1) ? "+" : ("+" + std::to_string(charge));
    } else if (charge < 0) {
        symbol += (charge == -1) ? "-" : std::to_string(charge);
    }
    const int64_t chiral = feat[3];
    if (chiral == 1) {
        symbol += "@@";
    } else if (chiral == 2) {
        symbol += "@";
    } else if (chiral > 2) {
        symbol += "@" + std::to_string(chiral);
    }
    return symbol;
}

template <typename Map, typename Key>
int64_t count_or_zero(const Map &m, const Key &key) {
    auto it = m.find(key);
    return it == m.end() ? 0 : it->second;
}

struct DfsPlan {
    std::vector<int> roots;
    std::vector<std::vector<std::pair<int, int>>> tree_children;
    std::vector<int> non_tree_edges;
    std::vector<int> entry_order;
    std::vector<int> old_to_new;
};

class FastStackKGuide {
  public:
    FastStackKGuide(
        py::list node_count_items,
        py::list edge_count_items,
        py::list triplet_count_items,
        py::object default_edge_obj,
        py::list node_vocab_items,
        py::list edge_vocab_items,
        int paren_open_id,
        int paren_close_id,
        int sep_id,
        int k_base,
        int k_min,
        bool omit_default_edge
    ) : default_edge_(parse_edge_key(default_edge_obj)),
        paren_open_id_(paren_open_id),
        paren_close_id_(paren_close_id),
        sep_id_(sep_id),
        k_base_(k_base),
        k_min_(k_min),
        omit_default_edge_(omit_default_edge) {
        node_counts_.reserve(py::len(node_count_items) * 2 + 1);
        for (py::handle item_h : node_count_items) {
            py::tuple item = py::reinterpret_borrow<py::tuple>(item_h);
            if (py::len(item) != 2) throw std::runtime_error("bad node count item");
            node_counts_[parse_node_key(item[0])] = py::cast<int64_t>(item[1]);
        }
        edge_counts_.reserve(py::len(edge_count_items) * 2 + 1);
        for (py::handle item_h : edge_count_items) {
            py::tuple item = py::reinterpret_borrow<py::tuple>(item_h);
            if (py::len(item) != 2) throw std::runtime_error("bad edge count item");
            edge_counts_[parse_edge_key(item[0])] = py::cast<int64_t>(item[1]);
        }
        triplet_counts_.reserve(py::len(triplet_count_items) * 2 + 1);
        for (py::handle item_h : triplet_count_items) {
            py::tuple item = py::reinterpret_borrow<py::tuple>(item_h);
            if (py::len(item) != 4) throw std::runtime_error("bad triplet count item");
            TripletKey key = make_triplet_key(parse_node_key(item[0]), parse_edge_key(item[1]), parse_node_key(item[2]));
            triplet_counts_[key] = py::cast<int64_t>(item[3]);
        }
        node_vocab_.reserve(py::len(node_vocab_items) * 2 + 1);
        for (py::handle item_h : node_vocab_items) {
            py::tuple item = py::reinterpret_borrow<py::tuple>(item_h);
            if (py::len(item) != 2) throw std::runtime_error("bad node vocab item");
            node_vocab_[parse_node_key(item[0])] = py::cast<int>(item[1]);
        }
        edge_vocab_.reserve(py::len(edge_vocab_items) * 2 + 1);
        for (py::handle item_h : edge_vocab_items) {
            py::tuple item = py::reinterpret_borrow<py::tuple>(item_h);
            if (py::len(item) != 2) throw std::runtime_error("bad edge vocab item");
            edge_vocab_[parse_edge_key(item[0])] = py::cast<int>(item[1]);
        }
    }

    void set_frequency_ordering(bool enabled) { frequency_ordering_ = enabled; }

    py::dict serialize(
        int num_nodes,
        py::array_t<int64_t, py::array::c_style | py::array::forcecast> src_arr,
        py::array_t<int64_t, py::array::c_style | py::array::forcecast> dst_arr,
        py::array_t<int64_t, py::array::c_style | py::array::forcecast> node_feat_arr,
        py::array_t<int64_t, py::array::c_style | py::array::forcecast> edge_feat_arr
    ) const {
        auto src = src_arr.unchecked<1>();
        auto dst = dst_arr.unchecked<1>();
        auto nfeat_raw = node_feat_arr.unchecked<2>();
        auto efeat_raw = edge_feat_arr.unchecked<2>();
        if (src.shape(0) != dst.shape(0) || src.shape(0) != efeat_raw.shape(0)) {
            throw std::runtime_error("src/dst/edge_features length mismatch");
        }
        if (nfeat_raw.shape(0) != num_nodes || nfeat_raw.shape(1) != 6 || efeat_raw.shape(1) != 3) {
            throw std::runtime_error("bad node_features or edge_features shape");
        }

        std::vector<NodeKey> nfeat(static_cast<size_t>(num_nodes));
        for (int i = 0; i < num_nodes; ++i) {
            for (int j = 0; j < 6; ++j) nfeat[i][j] = nfeat_raw(i, j);
        }

        std::vector<UndirectedEdge> edges;
        std::vector<std::vector<std::pair<int, int>>> adjacency;
        build_undirected_edges(num_nodes, src, dst, efeat_raw, edges, adjacency);
        DfsPlan plan = make_dfs_plan(num_nodes, nfeat, edges, adjacency);
        return emit_stackk(num_nodes, nfeat, edges, plan);
    }

    py::dict serialize_fddfs(
        int num_nodes,
        py::array_t<int64_t, py::array::c_style | py::array::forcecast> src_arr,
        py::array_t<int64_t, py::array::c_style | py::array::forcecast> dst_arr,
        py::array_t<int64_t, py::array::c_style | py::array::forcecast> node_feat_arr,
        py::array_t<int64_t, py::array::c_style | py::array::forcecast> edge_feat_arr,
        int ring_base,
        int ring_min
    ) const {
        auto src = src_arr.unchecked<1>();
        auto dst = dst_arr.unchecked<1>();
        auto nfeat_raw = node_feat_arr.unchecked<2>();
        auto efeat_raw = edge_feat_arr.unchecked<2>();
        if (src.shape(0) != dst.shape(0) || src.shape(0) != efeat_raw.shape(0)) {
            throw std::runtime_error("src/dst/edge_features length mismatch");
        }
        if (nfeat_raw.shape(0) != num_nodes || nfeat_raw.shape(1) != 6 || efeat_raw.shape(1) != 3) {
            throw std::runtime_error("bad node_features or edge_features shape");
        }
        if (ring_min < 1 || ring_base < 0) {
            throw std::runtime_error("bad FDDFS ring vocab parameters");
        }

        std::vector<NodeKey> nfeat(static_cast<size_t>(num_nodes));
        for (int i = 0; i < num_nodes; ++i) {
            for (int j = 0; j < 6; ++j) nfeat[i][j] = nfeat_raw(i, j);
        }

        std::vector<UndirectedEdge> edges;
        std::vector<std::vector<std::pair<int, int>>> adjacency;
        build_undirected_edges(num_nodes, src, dst, efeat_raw, edges, adjacency);
        DfsPlan plan = make_dfs_plan(num_nodes, nfeat, edges, adjacency);
        return emit_fddfs(num_nodes, nfeat, edges, plan, ring_base, ring_min);
    }

  private:
    std::unordered_map<NodeKey, int64_t, NodeHash> node_counts_;
    std::unordered_map<EdgeKey, int64_t, EdgeHash> edge_counts_;
    std::unordered_map<TripletKey, int64_t, TripletHash> triplet_counts_;
    std::unordered_map<NodeKey, int, NodeHash> node_vocab_;
    std::unordered_map<EdgeKey, int, EdgeHash> edge_vocab_;
    EdgeKey default_edge_{};
    int paren_open_id_ = 0;
    int paren_close_id_ = 1;
    int sep_id_ = 2;
    int k_base_ = 0;
    int k_min_ = 2;
    bool omit_default_edge_ = true;
    bool frequency_ordering_ = true;

    TransitionKey transition_key(
        const std::vector<NodeKey> &nfeat,
        int src_node,
        const UndirectedEdge &edge,
        int dst_node
    ) const {
        const NodeKey &src_feat = nfeat[static_cast<size_t>(src_node)];
        const NodeKey &dst_feat = nfeat[static_cast<size_t>(dst_node)];
        return TransitionKey{
            frequency_ordering_ ? count_or_zero(triplet_counts_, make_triplet_key(src_feat, edge.feat, dst_feat)) : 0,
            frequency_ordering_ ? count_or_zero(edge_counts_, edge.feat) : 0,
            frequency_ordering_ ? count_or_zero(node_counts_, dst_feat) : 0,
            compact_node_label(dst_feat),
            -dst_node,
        };
    }

    RootKey root_key(const std::vector<NodeKey> &nfeat, int node, int degree) const {
        const NodeKey &feat = nfeat[static_cast<size_t>(node)];
        return RootKey{
            frequency_ordering_ ? count_or_zero(node_counts_, feat) : 0,
            degree,
            compact_node_label(feat),
            -node,
        };
    }

    template <typename EFeatView>
    void build_undirected_edges(
        int num_nodes,
        const py::detail::unchecked_reference<int64_t, 1> &src,
        const py::detail::unchecked_reference<int64_t, 1> &dst,
        const EFeatView &efeat_raw,
        std::vector<UndirectedEdge> &edges,
        std::vector<std::vector<std::pair<int, int>>> &adjacency
    ) const {
        std::map<std::pair<int, int>, EdgeKey> by_pair;
        const ssize_t m = src.shape(0);
        for (ssize_t i = 0; i < m; ++i) {
            const int u = static_cast<int>(src(i));
            const int v = static_cast<int>(dst(i));
            if (u == v) throw std::runtime_error("self-loop is not supported by stackK");
            if (u < 0 || v < 0 || u >= num_nodes || v >= num_nodes) {
                throw std::runtime_error("edge endpoint out of range");
            }
            const std::pair<int, int> key = (u < v) ? std::make_pair(u, v) : std::make_pair(v, u);
            EdgeKey feat{};
            for (int j = 0; j < 3; ++j) feat[j] = efeat_raw(i, j);
            auto it = by_pair.find(key);
            if (it != by_pair.end() && it->second != feat) {
                throw std::runtime_error("asymmetric directed edge feature");
            }
            by_pair[key] = feat;
        }

        edges.reserve(by_pair.size());
        adjacency.assign(static_cast<size_t>(num_nodes), {});
        int edge_id = 0;
        for (const auto &kv : by_pair) {
            UndirectedEdge edge;
            edge.edge_id = edge_id;
            edge.u = kv.first.first;
            edge.v = kv.first.second;
            edge.feat = kv.second;
            edges.push_back(edge);
            adjacency[static_cast<size_t>(edge.u)].push_back({edge.v, edge_id});
            adjacency[static_cast<size_t>(edge.v)].push_back({edge.u, edge_id});
            ++edge_id;
        }
    }

    std::vector<std::vector<int>> connected_components(
        int num_nodes,
        const std::vector<std::vector<std::pair<int, int>>> &adjacency
    ) const {
        std::vector<char> seen(static_cast<size_t>(num_nodes), 0);
        std::vector<std::vector<int>> comps;
        for (int start = 0; start < num_nodes; ++start) {
            if (seen[static_cast<size_t>(start)]) continue;
            std::vector<int> stack{start};
            seen[static_cast<size_t>(start)] = 1;
            std::vector<int> comp;
            while (!stack.empty()) {
                int node = stack.back();
                stack.pop_back();
                comp.push_back(node);
                for (const auto &item : adjacency[static_cast<size_t>(node)]) {
                    int nbr = item.first;
                    if (!seen[static_cast<size_t>(nbr)]) {
                        seen[static_cast<size_t>(nbr)] = 1;
                        stack.push_back(nbr);
                    }
                }
            }
            std::sort(comp.begin(), comp.end());
            comps.push_back(std::move(comp));
        }
        return comps;
    }

    DfsPlan make_dfs_plan(
        int num_nodes,
        const std::vector<NodeKey> &nfeat,
        const std::vector<UndirectedEdge> &edges,
        const std::vector<std::vector<std::pair<int, int>>> &adjacency
    ) const {
        DfsPlan plan;
        plan.tree_children.assign(static_cast<size_t>(num_nodes), {});
        plan.entry_order.assign(static_cast<size_t>(num_nodes), -1);
        plan.old_to_new.assign(static_cast<size_t>(num_nodes), -1);
        std::vector<int> degree(static_cast<size_t>(num_nodes), 0);
        for (int i = 0; i < num_nodes; ++i) degree[static_cast<size_t>(i)] = static_cast<int>(adjacency[static_cast<size_t>(i)].size());

        std::vector<int> colors(static_cast<size_t>(num_nodes), 0);
        std::vector<char> tree_edge_ids(edges.size(), 0);
        std::vector<char> non_tree_edge_ids(edges.size(), 0);
        int next_entry_order = 0;
        int next_old_to_new = 0;

        auto sorted_neighbors = [&](int node) {
            std::vector<std::pair<int, int>> items = adjacency[static_cast<size_t>(node)];
            std::stable_sort(items.begin(), items.end(), [&](const auto &a, const auto &b) {
                return transition_greater(
                    transition_key(nfeat, node, edges[static_cast<size_t>(a.second)], a.first),
                    transition_key(nfeat, node, edges[static_cast<size_t>(b.second)], b.first)
                );
            });
            return items;
        };

        struct DfsFrame {
            int node = -1;
            int parent_edge = -1;
            std::vector<std::pair<int, int>> neighbors;
            size_t next_idx = 0;
        };

        auto enter_frame = [&](int node, int parent_edge) {
            colors[static_cast<size_t>(node)] = 1;
            plan.entry_order[static_cast<size_t>(node)] = next_entry_order++;
            plan.old_to_new[static_cast<size_t>(node)] = next_old_to_new++;

            auto neighbors = sorted_neighbors(node);
            if (neighbors.size() > 1) {
                std::rotate(neighbors.begin(), neighbors.begin() + 1, neighbors.end());
            }
            return DfsFrame{node, parent_edge, std::move(neighbors), 0};
        };

        auto run_dfs = [&](int root) {
            std::vector<DfsFrame> stack;
            stack.reserve(1024);
            stack.push_back(enter_frame(root, -1));
            while (!stack.empty()) {
                DfsFrame &frame = stack.back();
                if (frame.next_idx >= frame.neighbors.size()) {
                    colors[static_cast<size_t>(frame.node)] = 2;
                    stack.pop_back();
                    continue;
                }
                const auto item = frame.neighbors[frame.next_idx++];
                const int nbr = item.first;
                const int edge_id = item.second;
                if (frame.parent_edge >= 0 && edge_id == frame.parent_edge) continue;
                if (colors[static_cast<size_t>(nbr)] == 0) {
                    tree_edge_ids[static_cast<size_t>(edge_id)] = 1;
                    plan.tree_children[static_cast<size_t>(frame.node)].push_back({nbr, edge_id});
                    stack.push_back(enter_frame(nbr, edge_id));
                } else if (!tree_edge_ids[static_cast<size_t>(edge_id)]) {
                    non_tree_edge_ids[static_cast<size_t>(edge_id)] = 1;
                }
            }
        };

        std::vector<int> comp_roots;
        for (const auto &comp : connected_components(num_nodes, adjacency)) {
            int best = comp.front();
            RootKey best_key = root_key(nfeat, best, degree[static_cast<size_t>(best)]);
            for (int node : comp) {
                RootKey rk = root_key(nfeat, node, degree[static_cast<size_t>(node)]);
                if (root_greater(rk, best_key)) {
                    best = node;
                    best_key = std::move(rk);
                }
            }
            comp_roots.push_back(best);
        }
        std::stable_sort(comp_roots.begin(), comp_roots.end(), [&](int a, int b) {
            return root_greater(
                root_key(nfeat, a, degree[static_cast<size_t>(a)]),
                root_key(nfeat, b, degree[static_cast<size_t>(b)])
            );
        });

        for (int root : comp_roots) {
            if (colors[static_cast<size_t>(root)] == 0) {
                plan.roots.push_back(root);
                run_dfs(root);
            }
        }
        for (int node = 0; node < num_nodes; ++node) {
            if (colors[static_cast<size_t>(node)] == 0) {
                plan.roots.push_back(node);
                run_dfs(node);
            }
        }

        for (size_t edge_id = 0; edge_id < non_tree_edge_ids.size(); ++edge_id) {
            if (non_tree_edge_ids[edge_id]) plan.non_tree_edges.push_back(static_cast<int>(edge_id));
        }
        std::sort(plan.non_tree_edges.begin(), plan.non_tree_edges.end(), [&](int a, int b) {
            const auto &ea = edges[static_cast<size_t>(a)];
            const auto &eb = edges[static_cast<size_t>(b)];
            const int au = plan.entry_order[static_cast<size_t>(ea.u)];
            const int av = plan.entry_order[static_cast<size_t>(ea.v)];
            const int bu = plan.entry_order[static_cast<size_t>(eb.u)];
            const int bv = plan.entry_order[static_cast<size_t>(eb.v)];
            if (au != bu) return au < bu;
            if (av != bv) return av < bv;
            return a < b;
        });
        return plan;
    }

    void append_edge_token(std::vector<int> &ids, const EdgeKey &feat) const {
        if (omit_default_edge_ && feat == default_edge_) return;
        auto it = edge_vocab_.find(feat);
        if (it == edge_vocab_.end()) throw std::runtime_error("edge feature missing from vocab");
        ids.push_back(it->second);
    }

py::dict emit_stackk(
        int num_nodes,
        const std::vector<NodeKey> &nfeat,
        const std::vector<UndirectedEdge> &edges,
        const DfsPlan &plan
    ) const {
        std::vector<std::vector<int>> close_events(static_cast<size_t>(num_nodes));
        for (int edge_id : plan.non_tree_edges) {
            const auto &edge = edges[static_cast<size_t>(edge_id)];
            const int later = plan.entry_order[static_cast<size_t>(edge.u)] > plan.entry_order[static_cast<size_t>(edge.v)] ? edge.u : edge.v;
            close_events[static_cast<size_t>(later)].push_back(edge_id);
        }
        for (int node = 0; node < num_nodes; ++node) {
            auto &events = close_events[static_cast<size_t>(node)];
            std::stable_sort(events.begin(), events.end(), [&](int a, int b) {
                const auto &ea = edges[static_cast<size_t>(a)];
                const auto &eb = edges[static_cast<size_t>(b)];
                return transition_greater(
                    transition_key(nfeat, node, ea, ea.other(node)),
                    transition_key(nfeat, node, eb, eb.other(node))
                );
            });
        }

        std::vector<int> ids;
        ids.reserve(static_cast<size_t>(num_nodes) * 2);
        std::vector<int> current_stack;
        std::vector<int> stack_pos(static_cast<size_t>(num_nodes), -1);

        int64_t node_tokens = 0;
        int64_t non_tree_edges = 0;
        int64_t stackk_tokens = 0;
        int64_t sum_stackk_cycle_len = 0;
        int64_t branch_pairs = 0;
        int64_t component_sep = 0;
        int64_t max_stackk_cycle_len = 0;

        auto emit_node = [&](int node) {
            auto vit = node_vocab_.find(nfeat[static_cast<size_t>(node)]);
            if (vit == node_vocab_.end()) throw std::runtime_error("node feature missing from vocab");
            ids.push_back(vit->second);
            ++node_tokens;
            current_stack.push_back(node);
            stack_pos[static_cast<size_t>(node)] = static_cast<int>(current_stack.size()) - 1;
            for (int edge_id : close_events[static_cast<size_t>(node)]) {
                const auto &edge = edges[static_cast<size_t>(edge_id)];
                const int target = edge.other(node);
                const int target_pos = stack_pos[static_cast<size_t>(target)];
                if (target_pos < 0 || target_pos >= static_cast<int>(current_stack.size()) - 1) {
                    throw std::runtime_error("stackK non-tree edge target is not on current DFS stack");
                }
                const int cycle_len = static_cast<int>(current_stack.size()) - target_pos;
                if (cycle_len < 2) throw std::runtime_error("bad stackK cycle length");
                append_edge_token(ids, edge.feat);
                ids.push_back(k_base_ + cycle_len - k_min_);
                ++non_tree_edges;
                ++stackk_tokens;
                sum_stackk_cycle_len += cycle_len;
                if (cycle_len > max_stackk_cycle_len) max_stackk_cycle_len = cycle_len;
            }
        };

        enum class EmitActionType { Enter, Exit, OpenParen, CloseParen, EdgeToken };
        struct EmitAction {
            EmitActionType type;
            int node = -1;
            int edge_id = -1;
        };

        auto run_emit_subtree = [&](int root) {
            std::vector<EmitAction> actions;
            actions.reserve(1024);
            actions.push_back({EmitActionType::Enter, root, -1});
            while (!actions.empty()) {
                EmitAction action = actions.back();
                actions.pop_back();
                switch (action.type) {
                    case EmitActionType::OpenParen:
                        ids.push_back(paren_open_id_);
                        break;
                    case EmitActionType::CloseParen:
                        ids.push_back(paren_close_id_);
                        ++branch_pairs;
                        break;
                    case EmitActionType::EdgeToken:
                        append_edge_token(ids, edges[static_cast<size_t>(action.edge_id)].feat);
                        break;
                    case EmitActionType::Exit:
                        if (current_stack.empty() || current_stack.back() != action.node) {
                            throw std::runtime_error("stackK emit stack mismatch");
                        }
                        stack_pos[static_cast<size_t>(action.node)] = -1;
                        current_stack.pop_back();
                        break;
                    case EmitActionType::Enter: {
                        const int node = action.node;
                        emit_node(node);
                        const auto &children = plan.tree_children[static_cast<size_t>(node)];
                        actions.push_back({EmitActionType::Exit, node, -1});
                        if (!children.empty()) {
                            const auto &main_child = children.back();
                            actions.push_back({EmitActionType::Enter, main_child.first, -1});
                            actions.push_back({EmitActionType::EdgeToken, -1, main_child.second});
                            for (size_t idx = children.size() - 1; idx > 0; --idx) {
                                const auto &branch_child = children[idx - 1];
                                actions.push_back({EmitActionType::CloseParen, -1, -1});
                                actions.push_back({EmitActionType::Enter, branch_child.first, -1});
                                actions.push_back({EmitActionType::EdgeToken, -1, branch_child.second});
                                actions.push_back({EmitActionType::OpenParen, -1, -1});
                            }
                        }
                        break;
                    }
                }
            }
        };

        for (size_t i = 0; i < plan.roots.size(); ++i) {
            if (i != 0) {
                ids.push_back(sep_id_);
                ++component_sep;
            }
            run_emit_subtree(plan.roots[i]);
        }

        int64_t tree_edges = 0;
        for (const auto &children : plan.tree_children) tree_edges += static_cast<int64_t>(children.size());

        py::dict stats;
        stats["node_tokens"] = node_tokens;
        stats["non_tree_edges"] = non_tree_edges;
        stats["stackk_tokens"] = stackk_tokens;
        stats["sum_stackk_cycle_len"] = sum_stackk_cycle_len;
        stats["branch_pairs"] = branch_pairs;
        stats["component_sep"] = component_sep;
        stats["tokens"] = static_cast<int64_t>(ids.size());
        stats["tree_edges"] = tree_edges;
        stats["max_stackk_cycle_len"] = max_stackk_cycle_len;
        if (stackk_tokens > 0) {
            stats["mean_stackk_cycle_len_x1000"] = static_cast<int64_t>((sum_stackk_cycle_len * 1000.0 / stackk_tokens) + 0.5);
        }

        py::dict out;
        out["ids"] = std::move(ids);
        out["old_to_new"] = plan.old_to_new;
        out["stats"] = stats;
        return out;
    }

    py::dict emit_fddfs(
        int num_nodes,
        const std::vector<NodeKey> &nfeat,
        const std::vector<UndirectedEdge> &edges,
        const DfsPlan &plan,
        int ring_base,
        int ring_min
    ) const {
        struct RingEvent {
            int kind = 0;  // 0 = open, 1 = close
            int edge_id = -1;
        };

        std::vector<std::vector<RingEvent>> ring_events(static_cast<size_t>(num_nodes));
        for (int edge_id : plan.non_tree_edges) {
            const auto &edge = edges[static_cast<size_t>(edge_id)];
            int opener = edge.u;
            int closer = edge.v;
            if (plan.entry_order[static_cast<size_t>(edge.u)] > plan.entry_order[static_cast<size_t>(edge.v)]) {
                opener = edge.v;
                closer = edge.u;
            }
            ring_events[static_cast<size_t>(opener)].push_back({0, edge_id});
            ring_events[static_cast<size_t>(closer)].push_back({1, edge_id});
        }

        for (int node = 0; node < num_nodes; ++node) {
            auto &events = ring_events[static_cast<size_t>(node)];
            std::stable_sort(events.begin(), events.end(), [&](const RingEvent &a, const RingEvent &b) {
                if (a.kind != b.kind) return a.kind < b.kind;  // 同一节点先 open 后 close。
                const auto &ea = edges[static_cast<size_t>(a.edge_id)];
                const auto &eb = edges[static_cast<size_t>(b.edge_id)];
                return transition_greater(
                    transition_key(nfeat, node, ea, ea.other(node)),
                    transition_key(nfeat, node, eb, eb.other(node))
                );
            });
        }

        std::vector<int> ids;
        ids.reserve(static_cast<size_t>(num_nodes) * 2);
        std::vector<int> active(edges.size(), -1);
        std::vector<int> free_ids;
        int next_ring_id = ring_min;
        int max_ring_id = 0;
        int max_active_ring_ids = 0;
        int active_count = 0;

        int64_t node_tokens = 0;
        int64_t ring_open_tokens = 0;
        int64_t ring_close_tokens = 0;
        int64_t ring_tokens = 0;
        int64_t branch_pairs = 0;
        int64_t component_sep = 0;

        auto ring_token_id = [&](int ring_id) -> int {
            return ring_base + ring_id - ring_min;
        };

        auto allocate_ring_id = [&]() -> int {
            int rid = -1;
            if (!free_ids.empty()) {
                auto it = std::min_element(free_ids.begin(), free_ids.end());
                rid = *it;
                free_ids.erase(it);
            } else {
                rid = next_ring_id++;
            }
            if (rid > max_ring_id) max_ring_id = rid;
            return rid;
        };

        auto emit_node = [&](int node) {
            auto vit = node_vocab_.find(nfeat[static_cast<size_t>(node)]);
            if (vit == node_vocab_.end()) throw std::runtime_error("node feature missing from vocab");
            ids.push_back(vit->second);
            ++node_tokens;
            for (const RingEvent &event : ring_events[static_cast<size_t>(node)]) {
                const auto &edge = edges[static_cast<size_t>(event.edge_id)];
                if (event.kind == 0) {
                    const int rid = allocate_ring_id();
                    active[static_cast<size_t>(event.edge_id)] = rid;
                    append_edge_token(ids, edge.feat);
                    ids.push_back(ring_token_id(rid));
                    ++active_count;
                    if (active_count > max_active_ring_ids) max_active_ring_ids = active_count;
                    ++ring_open_tokens;
                    ++ring_tokens;
                } else {
                    const int rid = active[static_cast<size_t>(event.edge_id)];
                    if (rid < ring_min) throw std::runtime_error("FDDFS ring edge closed before open");
                    active[static_cast<size_t>(event.edge_id)] = -1;
                    ids.push_back(ring_token_id(rid));
                    free_ids.push_back(rid);
                    --active_count;
                    ++ring_close_tokens;
                    ++ring_tokens;
                }
            }
        };

        enum class EmitActionType { Enter, OpenParen, CloseParen, EdgeToken };
        struct EmitAction {
            EmitActionType type;
            int node = -1;
            int edge_id = -1;
        };

        auto run_emit_subtree = [&](int root) {
            std::vector<EmitAction> actions;
            actions.reserve(1024);
            actions.push_back({EmitActionType::Enter, root, -1});
            while (!actions.empty()) {
                EmitAction action = actions.back();
                actions.pop_back();
                switch (action.type) {
                    case EmitActionType::OpenParen:
                        ids.push_back(paren_open_id_);
                        break;
                    case EmitActionType::CloseParen:
                        ids.push_back(paren_close_id_);
                        ++branch_pairs;
                        break;
                    case EmitActionType::EdgeToken:
                        append_edge_token(ids, edges[static_cast<size_t>(action.edge_id)].feat);
                        break;
                    case EmitActionType::Enter: {
                        const int node = action.node;
                        emit_node(node);
                        const auto &children = plan.tree_children[static_cast<size_t>(node)];
                        if (!children.empty()) {
                            const auto &main_child = children.back();
                            actions.push_back({EmitActionType::Enter, main_child.first, -1});
                            actions.push_back({EmitActionType::EdgeToken, -1, main_child.second});
                            for (size_t idx = children.size() - 1; idx > 0; --idx) {
                                const auto &branch_child = children[idx - 1];
                                actions.push_back({EmitActionType::CloseParen, -1, -1});
                                actions.push_back({EmitActionType::Enter, branch_child.first, -1});
                                actions.push_back({EmitActionType::EdgeToken, -1, branch_child.second});
                                actions.push_back({EmitActionType::OpenParen, -1, -1});
                            }
                        }
                        break;
                    }
                }
            }
        };

        for (size_t i = 0; i < plan.roots.size(); ++i) {
            if (i != 0) {
                ids.push_back(sep_id_);
                ++component_sep;
            }
            run_emit_subtree(plan.roots[i]);
        }
        if (active_count != 0) throw std::runtime_error("FDDFS has unclosed ring ids");

        int64_t tree_edges = 0;
        for (const auto &children : plan.tree_children) tree_edges += static_cast<int64_t>(children.size());

        py::dict stats;
        stats["node_tokens"] = node_tokens;
        stats["ring_open_tokens"] = ring_open_tokens;
        stats["ring_close_tokens"] = ring_close_tokens;
        stats["ring_tokens"] = ring_tokens;
        stats["branch_pairs"] = branch_pairs;
        stats["component_sep"] = component_sep;
        stats["tokens"] = static_cast<int64_t>(ids.size());
        stats["tree_edges"] = tree_edges;
        stats["non_tree_edges"] = static_cast<int64_t>(plan.non_tree_edges.size());
        stats["max_active_ring_ids"] = static_cast<int64_t>(max_active_ring_ids);
        stats["max_ring_id"] = static_cast<int64_t>(max_ring_id);

        py::dict out;
        out["ids"] = std::move(ids);
        out["old_to_new"] = plan.old_to_new;
        out["stats"] = stats;
        return out;
    }
};

py::tuple node_tuple(const NodeKey &key) {
    py::tuple out(6);
    for (size_t i = 0; i < 6; ++i) out[i] = key[i];
    return out;
}

py::tuple edge_tuple(const EdgeKey &key) {
    py::tuple out(3);
    for (size_t i = 0; i < 3; ++i) out[i] = key[i];
    return out;
}

class FastStackKCounter {
  public:
    FastStackKCounter() {
        node_counts_.reserve(256);
        edge_counts_.reserve(32);
        triplet_counts_.reserve(1024);
    }

    void add_graph(
        int num_nodes,
        py::array_t<int64_t, py::array::c_style | py::array::forcecast> src_arr,
        py::array_t<int64_t, py::array::c_style | py::array::forcecast> dst_arr,
        py::array_t<int64_t, py::array::c_style | py::array::forcecast> node_feat_arr,
        py::array_t<int64_t, py::array::c_style | py::array::forcecast> edge_feat_arr
    ) {
        auto src = src_arr.unchecked<1>();
        auto dst = dst_arr.unchecked<1>();
        auto nfeat_raw = node_feat_arr.unchecked<2>();
        auto efeat_raw = edge_feat_arr.unchecked<2>();
        if (src.shape(0) != dst.shape(0) || src.shape(0) != efeat_raw.shape(0)) {
            throw std::runtime_error("src/dst/edge_features length mismatch");
        }
        if (nfeat_raw.shape(0) != num_nodes || nfeat_raw.shape(1) != 6 || efeat_raw.shape(1) != 3) {
            throw std::runtime_error("bad node_features or edge_features shape");
        }

        std::vector<NodeKey> nfeat(static_cast<size_t>(num_nodes));
        for (int i = 0; i < num_nodes; ++i) {
            for (int j = 0; j < 6; ++j) nfeat[static_cast<size_t>(i)][j] = nfeat_raw(i, j);
            node_counts_[nfeat[static_cast<size_t>(i)]] += 1;
        }

        std::vector<UndirectedEdge> edges;
        build_undirected_edges_for_count(num_nodes, src, dst, efeat_raw, edges);
        stats_records_ += 1;
        stats_nodes_ += num_nodes;
        stats_directed_edges_ += src.shape(0);
        stats_undirected_edges_ += static_cast<int64_t>(edges.size());

        for (const auto &edge : edges) {
            edge_counts_[edge.feat] += 1;
            const NodeKey &a = nfeat[static_cast<size_t>(edge.u)];
            const NodeKey &b = nfeat[static_cast<size_t>(edge.v)];
            triplet_counts_[make_triplet_key(a, edge.feat, b)] += 1;
            triplet_counts_[make_triplet_key(b, edge.feat, a)] += 1;
        }
    }

    py::dict result() const {
        py::list node_items;
        for (const auto &kv : node_counts_) {
            node_items.append(py::make_tuple(node_tuple(kv.first), kv.second));
        }
        py::list edge_items;
        for (const auto &kv : edge_counts_) {
            edge_items.append(py::make_tuple(edge_tuple(kv.first), kv.second));
        }
        py::list triplet_items;
        for (const auto &kv : triplet_counts_) {
            NodeKey src{};
            EdgeKey edge{};
            NodeKey dst{};
            for (size_t i = 0; i < 6; ++i) src[i] = kv.first[i];
            for (size_t i = 0; i < 3; ++i) edge[i] = kv.first[6 + i];
            for (size_t i = 0; i < 6; ++i) dst[i] = kv.first[9 + i];
            triplet_items.append(py::make_tuple(node_tuple(src), edge_tuple(edge), node_tuple(dst), kv.second));
        }
        py::dict stats;
        stats["records"] = stats_records_;
        stats["nodes"] = stats_nodes_;
        stats["directed_edges"] = stats_directed_edges_;
        stats["undirected_edges"] = stats_undirected_edges_;

        py::dict out;
        out["node_counts"] = node_items;
        out["edge_counts"] = edge_items;
        out["triplet_counts"] = triplet_items;
        out["stats"] = stats;
        return out;
    }

  private:
    std::unordered_map<NodeKey, int64_t, NodeHash> node_counts_;
    std::unordered_map<EdgeKey, int64_t, EdgeHash> edge_counts_;
    std::unordered_map<TripletKey, int64_t, TripletHash> triplet_counts_;
    int64_t stats_records_ = 0;
    int64_t stats_nodes_ = 0;
    int64_t stats_directed_edges_ = 0;
    int64_t stats_undirected_edges_ = 0;

    template <typename EFeatView>
    void build_undirected_edges_for_count(
        int num_nodes,
        const py::detail::unchecked_reference<int64_t, 1> &src,
        const py::detail::unchecked_reference<int64_t, 1> &dst,
        const EFeatView &efeat_raw,
        std::vector<UndirectedEdge> &edges
    ) const {
        std::map<std::pair<int, int>, EdgeKey> by_pair;
        const ssize_t m = src.shape(0);
        for (ssize_t i = 0; i < m; ++i) {
            const int u = static_cast<int>(src(i));
            const int v = static_cast<int>(dst(i));
            if (u == v) throw std::runtime_error("self-loop is not supported by stackK");
            if (u < 0 || v < 0 || u >= num_nodes || v >= num_nodes) {
                throw std::runtime_error("edge endpoint out of range");
            }
            const std::pair<int, int> key = (u < v) ? std::make_pair(u, v) : std::make_pair(v, u);
            EdgeKey feat{};
            for (int j = 0; j < 3; ++j) feat[j] = efeat_raw(i, j);
            auto it = by_pair.find(key);
            if (it != by_pair.end() && it->second != feat) {
                throw std::runtime_error("asymmetric directed edge feature");
            }
            by_pair[key] = feat;
        }

        edges.reserve(by_pair.size());
        int edge_id = 0;
        for (const auto &kv : by_pair) {
            UndirectedEdge edge;
            edge.edge_id = edge_id++;
            edge.u = kv.first.first;
            edge.v = kv.first.second;
            edge.feat = kv.second;
            edges.push_back(edge);
        }
    }
};

}  // namespace

#ifndef FDDFS_MODULE_NAME
#define FDDFS_MODULE_NAME _stackk_cpp
#endif
PYBIND11_MODULE(FDDFS_MODULE_NAME, m) {
    m.doc() = "C++ stackK serializer core";
    py::class_<FastStackKGuide>(m, "FastStackKGuide")
        .def(py::init<
             py::list,
             py::list,
             py::list,
             py::object,
             py::list,
             py::list,
             int,
             int,
             int,
             int,
             int,
             bool>())
        .def("set_frequency_ordering", &FastStackKGuide::set_frequency_ordering)
        .def("serialize", &FastStackKGuide::serialize)
        .def("serialize_fddfs", &FastStackKGuide::serialize_fddfs);
    py::class_<FastStackKCounter>(m, "FastStackKCounter")
        .def(py::init<>())
        .def("add_graph", &FastStackKCounter::add_graph)
        .def("result", &FastStackKCounter::result);
}
