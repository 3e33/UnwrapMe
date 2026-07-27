// Copyright (C) 2023-2026 Elias V.
// SPDX-License-Identifier: GPL-3.0-only

#include "aStar.h"

#include "structures.h"
#include "utilities.h"

#include <xtensor/xarray.hpp>
#include <xtensor/xmath.hpp>
#include <xtensor/xadapt.hpp>

#include <algorithm>
#include <cmath>
#include <execution>
#include <functional>
#include <iterator>
#include <limits>
#include <memory>
#include <queue>
#include <set>
#include <unordered_set>
#include <utility>

//------------------------------------------------------------
// A*
//------------------------------------------------------------

template <typename T>
T getShortestPath(const BVertPtr& startVert,
                  const BVertPtr& endVert,
                  const T& bmEdges) {
  priority_queue<shared_ptr<Node>, vector<shared_ptr<Node>>, Node> fringes;

  set<size_t> visited;
  T pathEdges;

  auto node = make_shared<Node>();
  node->bmVert = startVert;
  fringes.push(node);

  while (!fringes.empty()) {
    auto currentNode = fringes.top();
    fringes.pop();

    if (currentNode->bmVert == endVert) {
      while (currentNode->parentNode != nullptr) {
        pathEdges.insert(currentNode->bmEdge);
        currentNode = currentNode->parentNode;
      }

      return pathEdges;
    }

    vector<BEdgePtr> edges;
    for (auto& edge : currentNode->bmVert->link_edges) {
      if (bmEdges.find(edge) == bmEdges.end()) {
        continue;
      }

      edges.push_back(edge);
    }

    if (edges.empty()) {
      return {};
    }

    vector<double> edgeLengths;
    transform(edges.begin(),
              edges.end(),
              back_inserter(edgeLengths),
              [](const auto& e) { return e->length; });

    vector<double> coordsTmp;
    for (const auto& e : edges) {
      auto otherCoords = getEdgeOtherVert(e, currentNode->bmVert)->co;
      coordsTmp.push_back(otherCoords(0));
      coordsTmp.push_back(otherCoords(1));
      coordsTmp.push_back(otherCoords(2));
    }
    xt::xarray<double> coords = xt::adapt(coordsTmp);
    coords.reshape({edges.size(), 3});

    auto finalDistances = xt::sqrt(xt::sum(xt::pow(endVert->co - coords, 2), 1));

    xt::xarray<double> costs = currentNode->travelledDistance +
                               xt::adapt(edgeLengths) + finalDistances;
    vector<shared_ptr<Node>> nodes;
    for (size_t i = 0; i < costs.size(); i++) {
      auto node = make_shared<Node>();
      node->bmVert = getEdgeOtherVert(edges.at(i), currentNode->bmVert);
      node->cost = costs.at(i);
      node->parentNode = currentNode;
      node->travelledDistance =
          currentNode->travelledDistance + edgeLengths.at(i);
      node->bmEdge = edges.at(i);

      nodes.push_back(node);
    }

    for (size_t i = 0; i < nodes.size(); i++) {
      if (visited.find(nodes.at(i)->bmEdge->index) == visited.end()) {
        fringes.push(nodes.at(i));
        visited.insert(nodes.at(i)->bmEdge->index);
      }
    }
  }

  return {};
}

template set<BEdgePtr>
getShortestPath(const BVertPtr&,
                const BVertPtr&,
                const set<BEdgePtr>&);

template unordered_set<BEdgePtr>
getShortestPath(const BVertPtr&,
                const BVertPtr&,
                const unordered_set<BEdgePtr>&);
