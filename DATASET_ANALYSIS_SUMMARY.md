# Dataset Analysis Summary

## Dataset: Fai Datasets Rishu.xlsx

### Basic Information
- **Number of rows:** 1,195
- **Number of columns:** 11
- **Sheet name:** Sheet1

### Column Names
1. Enter your age:
2. Which platform do you use the most daily?
3. On average, how much time do you spend on social media per day?
4. When do you use social media the most?
5. What do you primarily use social media for?
6. Have you ever tried to take breaks from social media?
7. Do you feel social media affects your mental health?
8. What type of content do you engage with the most?
9. Have you ever experienced online drama or conflict because of social media?
10. Do you trust influencers' product recommendations?
11. Would you ever delete social media permanently?

### Data Types
- **Numerical (1):** Enter your age: (int64)
- **Categorical (10):** All other columns (object)

### Missing Values
| Column | Missing Count | Missing % |
|--------|---------------|-----------|
| Which platform do you use the most daily? | 5 | 0.42% |
| On average, how much time do you spend on social media per day? | 1 | 0.08% |
| When do you use social media the most? | 2 | 0.17% |
| What do you primarily use social media for? | 2 | 0.17% |
| What type of content do you engage with the most? | 3 | 0.25% |
| Have you ever experienced online drama or conflict because of social media? | 1 | 0.08% |
| Do you trust influencers' product recommendations? | 1 | 0.08% |
| Would you ever delete social media permanently? | 1 | 0.08% |

**Total duplicate rows:** 0

### Age Statistics
- **Range:** 13 - 59 years
- **Mean:** 37.71 years
- **Median:** 38.00 years
- **Standard deviation:** 12.33 years

**Age group distribution:**
- 10-19: 92 (7.7%)
- 20-29: 284 (23.8%)
- 30-39: 272 (22.8%)
- 40-49: 291 (24.4%)
- 50-59: 256 (21.4%)

### Categorical Variable Distributions

#### 1. Platform Used (50 unique values - **DATA QUALITY ISSUE**)
Top 10 platforms:
- Instagram: 186 (15.6%)
- Disney+: 112 (9.4%)
- Reddit: 110 (9.2%)
- Netflix: 105 (8.8%)
- Twitch: 102 (8.5%)
- TikTok: 102 (8.5%)
- YouTube: 102 (8.5%)
- Snapchat: 101 (8.5%)
- Amazon Prime Video: 99 (8.3%)
- Spotify: 93 (7.8%)

**Issues:**
- Inconsistent naming (Instagram, instagram, INSTAGRAM, instagarm, Insta, etc.)
- Multiple platforms in single cells (e.g., "Instagram and Spotify")
- Non-social media entries (Wps office, Stocks, BGMI, platinum(Pt), Ground)

#### 2. Time Spent on Social Media (4 categories)
- 3-4 hours: 326 (27.3%)
- 1-2 hours: 302 (25.3%)
- 5+ hours: 293 (24.5%)
- Less than 1 hour: 273 (22.8%)

#### 3. Time of Day Usage (4 categories)
- Evening: 331 (27.7%)
- Late Night: 319 (26.7%)
- Afternoon: 284 (23.8%)
- Morning: 259 (21.7%)

#### 4. Primary Purpose (4 categories)
- Entertainment: 376 (31.5%)
- Chatting with friend/family: 291 (24.4%)
- Studying/Work-related tasks: 278 (23.3%)
- Shopping: 248 (20.8%)

#### 5. Break Attempts (4 categories)
- Yes: 348 (29.1%)
- Maybe: 298 (24.9%)
- No: 277 (23.2%)
- Thinking about it: 272 (22.8%)

#### 6. Mental Health Impact (3 categories)
- Yes: 422 (35.3%)
- No: 393 (32.9%)
- Maybe: 380 (31.8%)

#### 7. Content Type Engaged (4 categories)
- Reels: 396 (33.1%)
- Stories: 279 (23.3%)
- Posts: 267 (22.3%)
- Tweets: 250 (20.9%)

#### 8. Online Drama Experience (3 categories)
- No: 420 (35.1%)
- Yes: 397 (33.2%)
- Maybe: 377 (31.5%)

#### 9. Trust in Influencers (3 categories)
- No: 442 (37.0%)
- Sometimes: 393 (32.9%)
- Yes: 359 (30.0%)

#### 10. Delete Social Media Permanently (3 categories)
- No: 453 (37.9%)
- Maybe: 373 (31.2%)
- Yes: 368 (30.8%)

### Data Quality Issues
1. **Platform column has severe data quality problems:**
   - Inconsistent capitalization and spelling
   - Multiple platforms listed in single cells
   - Non-social media platforms included
   - Requires significant cleaning before use

2. **Missing values:** Low overall (< 0.5% per column), but need imputation strategy

3. **Survey data limitations:**
   - Self-reported responses
   - Potential social desirability bias
   - No geographic or demographic information
   - Cross-sectional (single time point)

### Potential Target Variables for ML Tasks

#### Classification Tasks
1. **Would you ever delete social media permanently?** (Yes/No/Maybe)
   - Balanced classes: 30.8% / 37.9% / 31.2%
   - Could predict likelihood of social media abandonment

2. **Do you feel social media affects your mental health?** (Yes/No/Maybe)
   - Balanced classes: 35.3% / 32.9% / 31.8%
   - Could predict mental health impact perception

3. **Have you ever experienced online drama or conflict because of social media?** (Yes/No/Maybe)
   - Balanced classes: 33.2% / 35.1% / 31.5%
   - Could predict conflict experience

4. **Do you trust influencers' product recommendations?** (Yes/Sometimes/No)
   - Balanced classes: 30.0% / 32.9% / 37.0%
   - Could predict trust in influencers

#### Regression Task
- **Age prediction** from behavioral patterns
- Could be useful for demographic profiling

### Potential Predictor Variables
- Age (numerical)
- Platform used (categorical - needs cleaning)
- Time spent on social media (ordinal: could encode as 0, 1, 2, 3)
- Time of day usage (categorical)
- Primary purpose (categorical)
- Break attempts (categorical)
- Content type engaged (categorical)
- Online drama experience (categorical)
- Trust in influencers (categorical)

### Dataset Suitability Assessment

#### Suitable For:
- **Classification tasks** predicting behavioral outcomes
- **Exploratory data analysis** of social media usage patterns
- **Pattern recognition** in user behavior
- **Feature engineering** practice

#### Not Suitable For:
- **Causal inference** (survey data, no experimental design)
- **Time-series analysis** (cross-sectional data)
- **Complex multi-class prediction** with platform as feature (due to data quality issues)
- **General population inference** (selection bias)

### Biases and Limitations

1. **Selection Bias:**
   - Survey respondents may not represent general population
   - People who take surveys about social media may have different attitudes

2. **Response Bias:**
   - Social desirability may affect answers (e.g., underreporting time spent)
   - Self-reported data may be inaccurate

3. **Demographic Bias:**
   - Age range limited to 13-59
   - No information about gender, location, socioeconomic status
   - Geographic distribution unknown

4. **Platform Bias:**
   - Platform column includes non-social media platforms
   - Streaming services (Netflix, Disney+) mixed with social media
   - Gaming platforms (BGMI, Free fire) included

5. **Sample Size:**
   - 1,195 responses is moderate but may limit complex modeling
   - Some categories have very few samples after platform cleaning

### What This Dataset Can Legitimately Be Used For

**Appropriate uses:**
1. Predicting whether users would delete social media based on usage patterns
2. Classifying perceived mental health impact from usage behavior
3. Exploring correlations between time spent and self-reported outcomes
4. Building simple classification models for behavioral prediction
5. Educational purposes for data preprocessing and ML pipeline development

**Inappropriate uses:**
1. Establishing causal relationships between social media and mental health
2. Making claims about general population behavior
3. Predicting specific platform adoption (due to data quality issues)
4. Inferring demographic characteristics not in the dataset
5. Policy recommendations based on this sample alone

### Recommendations for Research Paper Alignment

When you provide the research paper, I will:

1. **Identify the specific research question** from the paper
2. **Determine if this dataset can legitimately address** that question
3. **If suitable:** Design experiment following paper's methodology as closely as possible
4. **If not suitable:** Explain clearly why and propose scientifically valid adaptations
5. **Ensure no data leakage** between features and target
6. **Use appropriate evaluation metrics** for the specific task
7. **Clearly distinguish** between paper results and our implementation results

### Next Steps

**Please provide the research/survey paper.** Once I have the paper, I will:

1. Analyze the paper's methodology and requirements
2. Determine which experiments from the paper can be reproduced with this dataset
3. Design an appropriate experimental setup
4. Implement the solution with proper preprocessing
5. Evaluate and compare results with the paper
6. Create a complete, reproducible GitHub-ready project

### Visualizations Generated

The following visualizations have been created in the `figures/` directory:
1. `basic_distributions.png` - Age distribution, time spent, mental health impact, deletion intention
2. `platform_distribution.png` - Top 10 platforms used
3. `purpose_distribution.png` - Primary purpose of social media usage
4. `mental_health_vs_time.png` - Mental health impact by time spent on social media

These visualizations provide initial insights into the data structure and distributions.
