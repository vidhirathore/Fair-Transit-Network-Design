# Fair Transit Network Design

This repository contains the implementation and findings of the **Fair Transit Network Design** project. The project aims to create equitable public transit networks that serve all community members effectively, with a focus on reducing inequalities in service distribution and improving accessibility.

## Overview

Equity in public transit is crucial to providing fair access to essential services such as employment, education, and healthcare. This project explores innovative methodologies to address inequities in transit systems, incorporating advanced optimization techniques and fairness metrics.

Key features of the project include:
- Multi-objective optimization models integrating equity and efficiency.
- Stochastic optimization techniques to account for uncertainties in transit demand.
- Application of fairness metrics, including the Gini coefficient, to measure inequality.

## Highlights

### Results
- **Gini Coefficient Improvement**: Reduced from 0.3339 (baseline) to 0.0627 post-optimization, indicating significant improvements in service equity.
- **Service Level Enhancements**: Improved service in underserved areas, particularly for marginalized communities.
- **Optimized Transit Network**: Balanced routes and frequencies within budgetary constraints.

### Methodologies
1. **Enhanced Multi-Objective Optimization**:
   - Built on Mixed-Integer Linear Programming (MILP).
   - Incorporated equity constraints and metrics such as the Gini coefficient.
   - Utilized Pareto optimization techniques.
2. **Stochastic Optimization**:
   - Addressed demand uncertainties.
   - Ensured robust network performance across diverse scenarios.

### Implementation
- **Dataset**: Public transit data for Osaka, Japan, enriched with synthetic fairness attributes.
- **Optimization Tools**: NetworkX for graph structures, and PuLP for MILP modeling.
- **Visualization**: Detailed plots for service levels, Gini coefficients, and optimized network designs.

## Getting Started

### Prerequisites
- Python 3.8 or above
- Required libraries:
  - NetworkX
  - PuLP
  - Matplotlib


## Key Metrics
- **Equity**: Measured using the Gini coefficient.
- **Efficiency**: Evaluated by total network cost and passenger flow.
- **Robustness**: Ensured through stochastic modeling.

## Challenges
- High computational complexity of MILP models.
- Balancing efficiency and fairness in large-scale networks.

