#pragma once

#include <numeric>
#include <vector>


typedef enum { kLeft, kRight } Direction;


template <typename T>
class DecodingTree {
public:
    explicit DecodingTree(const std::vector<T>& codes, const std::vector<size_t>& counts)
        : tree_(1, Node())
        , position_(0)
    {
        if (std::accumulate(counts.begin(), counts.end(), 0) != codes.size()) {
            throw std::runtime_error("DecodingTree: inconsistent number of codes.");
        }
        
        int levelInTree = 0;
        int level = 0;
        int position = 0;
        size_t cumCounts = 0;
        
        for (size_t i = 0; i < codes.size(); ++i) {
            while (cumCounts == i) {
                cumCounts += counts[level];
                ++level;
            }
            
            const T& code = codes[i];
            while (levelInTree < level) {
                Node& node = tree_[position];
                int newPosition = tree_.size();
                if (node.left == Node::UNDEFINED) {
                    node.left = newPosition;
                } else if (node.right == Node::UNDEFINED) {
                    node.right = newPosition;
                } else {
                    position = node.parent;
                    --levelInTree;
                    if (levelInTree < 0) {
                        throw std::runtime_error("DecodingTree: incorrect tree definition.");
                    }
                    continue;
                }
                Node newNode;
                newNode.parent = position;
                tree_.push_back(newNode);
                position = newPosition;
                ++levelInTree;
            }
            Node& node = tree_[position];
            node.value = code;
            node.isLeaf = true;
            position = node.parent;
            --levelInTree;
        }
    }
    
    bool step(Direction dir) {
        if (dir == kLeft) {
            position_ = tree_[position_].left;
        } else {
            position_ = tree_[position_].right;
        }
        if (position_ >= tree_.size()) {
            throw std::runtime_error("DecodingTree: traversed out of bounds.");
        }
        return tree_[position_].isLeaf;
    }
    
    const T& getValue() const {
        const Node& node = tree_.at(position_);
        if (!node.isLeaf) {
            throw std::runtime_error("DecodingTree: taking value of a non-leaf node.");
        }
        return node.value;
    }
    
    void reset() { position_ = 0; }

private:
    struct Node {
        static const int UNDEFINED = -1;
        Node() : left(UNDEFINED), right(UNDEFINED), isLeaf(false) {}
        int left;
        int right;
        int parent;
        bool isLeaf;
        T value;
    };
    std::vector<Node> tree_;
    int position_;
};
