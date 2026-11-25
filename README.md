# Premier League Match Outcome Prediction

UCF Data Science I · ISC 4241 · Fall 2026

## Research question

Can a random forest predict Premier League match outcomes better than multinomial logistic regression using the same information available before kickoff?

Each row represents one completed league match. The target has three classes: **home win, draw, and away win**. The home and away assignments are part of the matchup.

## 1. Collect and explore the data

- Use [TheSportsDB season endpoint](https://www.thesportsdb.com/documentation) to collect historical Premier League matches. Record the seasons, retrieval date, and fields used.
- Verify that each selected season has complete fixture coverage before calculating team form. The documented free season endpoint lists a result limit. Check actual counts from the export.
- Plot class counts by season and inspect the score distribution. Use these plots to explain class imbalance and how the data changes over time.

## 2. Clean the data and build features

- Keep completed Premier League matches. Check match IDs for duplicates and investigate conflicting records. Parse kickoff dates and scores and check that team IDs are consistent.
- Log row counts before and after each cleaning step, including exclusions for missing scores, invalid dates, and insufficient team history. Verify unusually large scorelines against the match record and retain confirmed results.
- Calculate points per match and average goal difference over each team's previous five completed league matches. Start with home-minus-away differences. Use a match for modeling once both teams have five prior matches in the verified data and report how many rows this excludes.
- Derive the label from the current match's final score. Build inputs only from information available before kickoff. Check for redundant features such as goals for, goals against, and goal difference calculated over the same window.

## 3. Train and compare

| Approach | Why use it? |
| --- | --- |
| Most-common-outcome baseline | Shows how well we can do by always predicting the most frequent outcome in the training data. |
| Multinomial logistic regression | Provides a simple, interpretable model for three outcomes. |
| Random forest classifier | Tests whether nonlinear relationships and interactions improve predictions. |

1. Split chronologically by season. Use earlier seasons for training, the next complete season for validation, and the newest complete season for testing.
2. Fit any preprocessing on training data and choose model settings on validation data. Start with a small set of random forest depths and minimum leaf sizes.
3. Refit each model and its preprocessing on training plus validation with the chosen settings. Evaluate on the held-out test season. Earlier completed matches in that season can update rolling features for later fixtures while model parameters stay fixed.

## 4. Evaluate and present

- Report accuracy and macro F1 on the same test fixtures for every approach. Macro F1 gives equal weight to all three classes.
- Show a confusion matrix and discuss draw recall. Walk through one actual prediction and explain a mistake or a limitation.
- State what the results support. The project tests whether the forest improves on simpler approaches and how consistently it handles all three outcomes.

## Presentation plan

| Section | Time |
| --- | ---: |
| Research question, dataset, and exploration | 2 min |
| Cleaning, outliers, and one example feature row | 3 min |
| Model choices, season split, and refitting | 3 min |
| Results, mistakes, and limitations | 3 min |

Each group member makes and presents their own slides. Be ready to explain what you did, why you did it, and what the evidence shows. Keep the talk within 12 minutes, submit the slides before the scheduled presentation, and have every group member attend.
