### Prepare dataset
#### Download the collected raw data from [Here]([https://pages.github.com/](https://drive.google.com/file/d/153bKA1P7AaAtN6ituYQetku5FPed-g-5/view?usp=drive_link)) and decompress it and rename the folder to "data".

#### Preprocess the raw data

```
python3 data_cleaning.py
```

#### Classify repositories into AI and non-AI classes. The code will also output the class size & the corresponding average popularity scores.

```
python3 hypothesis_test.py
```


#### Run MLP training, testing, and Shapley Values analysis
- Line 19: Determine which dataset (All GitHub repos, AI-class repos, or non-AI class repos) to analyze.
- Line 20: Determine the model checkpoint name
- Line 21: Set to True if analyze only the top-3000 repos.

```
python3 mlp_shapley.py
```
