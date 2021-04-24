## Results

Result table from the paper

| Method | Restaurant | Laptop |
| --- | --- | --- |
| Majority | 0.65 | 53.45 | 
| LSTM | 0.743 | 0.665 |
| TD-LSTM | 0.756 | 0.681 |
| AE-LSTM | 0.762 | 0.689 |
| ATAE-LSTM | 0.772 | 0.687 |
| IAN | 0.786 | 0.721 |

Results from my implementation

| Method | Restaurant | Laptop | Match |
| --- | --- | --- | --- |
| Majority | 0.65 | 0.535 | **YES** | 
| LSTM | 0.7482 | 0.6888 | **YES** |
| TD-LSTM |  |  | |
| AE-LSTM |  |  | |
| ATAE-LSTM |  |  | |
| IAN |  |  | |

**NOTE**: The LSTM baseline is of questionable validity. If we don't consider aspect terms, the problem becomes a multi-label classification (for any given text, the sentiment labels corresponding to each aspect term are all valid labels for the text.) and not multi-class as formulated in the paper. 

**TODO**  

- [ ] Implement a multi-task LSTM baseline.

