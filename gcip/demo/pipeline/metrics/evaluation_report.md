### FR3 propensity: validation and test (calibrated LightGBM vs rules baseline)

| Product | Split | Customers | Take-ups | ROC-AUC model | ROC-AUC rules | PR-AUC model | PR-AUC rules | Brier | ECE | Lift top 10% | Above threshold | Precision | Recall |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| GP01 | validation | 3176 | 162 | 0.711 | 0.586 | 0.111 | 0.068 | 0.047 | 0.008 | 2.53 | 14.1% | 11.8% | 32.7% |
| GP01 | test | 3258 | 164 | 0.728 | 0.628 | 0.129 | 0.073 | 0.046 | 0.010 | 3.05 | 15.5% | 12.9% | 39.6% |
| GP02 | validation | 982 | 49 | 0.617 | 0.604 | 0.086 | 0.067 | 0.047 | 0.008 | 2.04 | 2.9% | 20.7% | 12.2% |
| GP02 | test | 1063 | 31 | 0.634 | 0.573 | 0.043 | 0.037 | 0.029 | 0.014 | 1.94 | 2.9% | 6.5% | 6.5% |
| GP03 | validation | 2172 | 31 | 0.705 | 0.602 | 0.037 | 0.020 | 0.014 | 0.008 | 2.91 | 12.9% | 3.9% | 35.5% |
| GP03 | test | 2222 | 56 | 0.726 | 0.623 | 0.067 | 0.039 | 0.024 | 0.006 | 3.40 | 12.6% | 7.5% | 37.5% |
| GP04 | validation | 937 | 114 | 0.852 | 0.755 | 0.519 | 0.258 | 0.077 | 0.023 | 4.98 | 16.2% | 48.0% | 64.0% |
| GP04 | test | 951 | 122 | 0.821 | 0.737 | 0.529 | 0.249 | 0.081 | 0.012 | 4.10 | 17.7% | 44.0% | 60.7% |
| GP05 | validation | 888 | 45 | 0.763 | 0.690 | 0.201 | 0.131 | 0.046 | 0.020 | 4.21 | 8.9% | 20.2% | 35.6% |
| GP05 | test | 883 | 60 | 0.744 | 0.677 | 0.237 | 0.162 | 0.056 | 0.013 | 4.35 | 10.3% | 28.6% | 43.3% |
| GP06 | validation | 3578 | 116 | 0.670 | 0.610 | 0.077 | 0.050 | 0.031 | 0.007 | 3.27 | 6.2% | 12.2% | 23.3% |
| GP06 | test | 3696 | 106 | 0.709 | 0.625 | 0.090 | 0.067 | 0.027 | 0.005 | 3.67 | 7.3% | 12.3% | 31.1% |

Mean over the six products:

| Split | Scorer | ROC-AUC | PR-AUC | Brier | ECE | Lift top 10% |
|---|---|---|---|---|---|---|
| validation | model | 0.720 | 0.172 | 0.044 | 0.012 | 3.33 |
| validation | rules baseline | 0.641 | 0.099 | 0.064 | 0.089 | 2.34 |
| test | model | 0.727 | 0.182 | 0.044 | 0.010 | 3.42 |
| test | rules baseline | 0.644 | 0.105 | 0.066 | 0.094 | 2.65 |

### Fairness at the suggestion threshold (groups of 100+ customers)

Selection rate = share of the group above the product's threshold; calibration gap = largest difference between a group's take-up rate and its mean propensity.

| Product | Attribute | Validation: selection rate (lowest-highest group) | Validation: calibration gap | Test: selection rate | Test: calibration gap |
|---|---|---|---|---|---|
| GP01 | age band | 11.9%-16.5% | 1.6% | 13.6%-17.7% | 0.7% |
| GP01 | declared income band | 0.0%-40.3% | 1.7% | 0.0%-42.4% | 2.0% |
| GP01 | region | 6.8%-21.0% | 2.0% | 6.5%-22.0% | 2.6% |
| GP01 | vulnerable | 10.8%-14.4% | 0.4% | 11.9%-15.8% | 0.7% |
| GP02 | age band | 1.9%-6.3% | 3.6% | 1.0%-5.4% | 3.5% |
| GP02 | declared income band | 0.8%-4.1% | 2.8% | 0.4%-4.9% | 3.6% |
| GP02 | region | 1.0%-5.0% | 2.1% | 0.7%-6.1% | 2.3% |
| GP02 | vulnerable | 2.8%-2.8% | 0.9% | 2.9%-2.9% | 1.3% |
| GP03 | age band | 10.0%-14.2% | 0.6% | 9.8%-14.0% | 0.9% |
| GP03 | declared income band | 0.0%-38.0% | 1.5% | 0.0%-34.1% | 1.0% |
| GP03 | region | 2.5%-23.9% | 1.0% | 1.9%-21.4% | 2.0% |
| GP03 | vulnerable | 12.3%-13.0% | 1.6% | 9.4%-12.9% | 0.7% |
| GP04 | age band | 5.9%-61.2% | 2.3% | 6.5%-59.2% | 1.9% |
| GP04 | declared income band | 10.3%-19.7% | 2.2% | 9.9%-23.5% | 4.7% |
| GP04 | region | 11.7%-21.4% | 2.6% | 11.8%-24.0% | 4.0% |
| GP04 | vulnerable | 16.2%-16.2% | 0.3% | 17.7%-17.7% | 0.5% |
| GP05 | age band | 5.3%-15.4% | 2.8% | 7.3%-17.5% | 1.8% |
| GP05 | declared income band | 3.5%-13.3% | 2.9% | 4.9%-18.1% | 2.7% |
| GP05 | region | 7.8%-13.5% | 3.3% | 7.2%-15.0% | 2.9% |
| GP05 | vulnerable | 8.9%-8.9% | 1.8% | 10.3%-10.3% | 0.1% |
| GP06 | age band | 2.8%-14.6% | 0.7% | 3.4%-14.5% | 1.1% |
| GP06 | declared income band | 3.4%-11.8% | 1.5% | 4.8%-14.1% | 1.4% |
| GP06 | region | 3.2%-8.6% | 2.1% | 4.8%-11.8% | 1.7% |
| GP06 | vulnerable | 6.2%-6.8% | 1.8% | 6.2%-7.4% | 0.4% |

### FR2 categoriser: validation and test transactions (full categoriser vs rules only)

| Slice | Transactions | Category accuracy | Macro F1 | Sub-category accuracy | Rules only: accuracy | Rules only: macro F1 |
|---|---|---|---|---|---|---|
| validation seen merchants | 3,376,996 | 99.0% | 0.925 | 99.0% | 93.1% | 0.779 |
| validation unseen merchants | 769,636 | 90.1% | 0.688 | 88.0% | 89.0% | 0.686 |
| test seen merchants | 3,492,020 | 99.0% | 0.925 | 99.0% | 93.1% | 0.780 |
| test unseen merchants | 802,377 | 90.4% | 0.688 | 88.3% | 89.4% | 0.692 |

Out of distribution: 68,000 real US bank descriptions, coarse agreement 82.1%.

### FR4 impact inputs vs the simulator's truth

| Input | Customers | Median ratio | Median abs. error | Within 20% | Correlation |
|---|---|---|---|---|---|
| fuel litres | 16,745 | 1.057 | 7.4% | 82.7% | 0.953 |
| gas kwh | 19,111 | 1.050 | 27.9% | 38.8% | 0.570 |
| electricity kwh | 24,880 | 0.939 | 29.9% | 38.8% | 0.775 |
| transport co2e kg | 16,745 | 1.057 | 7.4% | 82.7% | 0.955 |
| home energy co2e kg | 24,880 | 1.084 | 22.1% | 47.0% | 0.743 |

### FR5 explanations

Templates: 5,437 of 5,437 pass every guard. LLM rewrites (Qwen/Qwen2.5-1.5B-Instruct, local): 89.7% of 360 pass; by product GP01 91.7%, GP02 88.3%, GP03 96.7%, GP04 93.3%, GP05 93.3%, GP06 75.0%.

Gemini API (gemini-3.5-flash, gemini-3.1-flash-lite), the provider at inference: 12 of 12 rewrites pass every guard (sample, 2026-10-11 07:10).

### Targeting backtest (test customers, same number contacted by each strategy)

| Strategy | Contacted | Take-ups | Lifetime tCO2e | Bank value (£) |
|---|---|---|---|---|
| model | 1,343 | 221 | 1,682.2 | 117,149 |
| rules | 1,343 | 159 | 1,174.4 | 78,894 |
| random | 1,343 | 80 | 579.8 | 46,433 |
