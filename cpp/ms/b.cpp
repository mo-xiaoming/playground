struct Node {
    std::string value;
    std::vector<Node*> children;
};

std::vector<Node*> traverse(Node &node) {
    std::vector<Node *> all_reachable_nodes{};
    for (auto const &n : children) {
        auto const nodes = traverse(n);
        all_reachable_nodes.insert(all_reachable_nodes.end(), nodes.cbegin(), nodes.cend());
    }
    std::sort(children.begin(), children.end());
    std::sort(all_reachable_nodes.begin(), all_reachable_nodes.end());
    std::vector<Node*> result;
    std::set_difference(children.begin(), children.end(), all_reachable_nodes.begin(), all_reachable_nodes.end(), std::back_inserter(result));
    node.children = result;
}
