
<h3><br><br>Introduction</h3>
<p>
This is an in-depth data analysis project that goes through every step of the data analysis process. The end goal is to create the best possible prediction model for car price. Below I will go into detail on each of the four files in this project to highlight the main points.
</p><br><br>




<h4>Data Wrangling:</h4>
<p>In this notebook, the primary focus is on preparing and cleaning the dataset for further analysis. The main steps involved in the data wrangling process are:</p>
<ol>
<li>Importing necessary libraries: The required libraries, such as pandas and numpy, are imported to facilitate data manipulation and cleaning.</li>
<li>Loading the dataset: The raw dataset is loaded into a pandas DataFrame for easier manipulation and analysis.</li>
<li>Data inspection: The dataset's structure, dimensions, and basic statistics are examined to get an initial understanding of the data.</li>
<li>Handling missing values: Missing values are identified, and appropriate imputation or removal strategies are employed to ensure the dataset's integrity.</li>
<li>Data transformation: Certain columns are transformed or reformatted to facilitate further analysis, such as converting data types or standardizing data.</li>
<li>Removing duplicates: Duplicate records, if any, are identified and removed to ensure data consistency and accuracy.</li>
<li>Exporting the cleaned dataset: The cleaned and processed dataset is exported to a new file, which can be used for further analysis and modeling.</li>
</ol><br><br>


<h4>Exploratory Data Analysis (EDA):</h4>
<p>In this notebook, the primary focus is on gaining insights and understanding the underlying patterns within the dataset. The main steps involved in the exploratory data analysis process are:</p>
<ol>
<li>Importing necessary libraries: Required libraries, such as pandas, numpy, matplotlib, and seaborn, are imported to facilitate data visualization and exploration.</li>
<li>Loading the cleaned dataset: The cleaned dataset from the data wrangling step is loaded into a pandas DataFrame for further exploration and analysis.</li>
<li>Univariate analysis: The distribution of individual variables is examined using histograms, bar plots, and summary statistics to understand the central tendency, spread, and shape of each feature.</li>
<li>Bivariate analysis: Relationships between pairs of variables are explored using scatter plots, box plots, and correlation analysis to identify potential associations and trends.</li>
<li>Multivariate analysis: Interactions among multiple variables are investigated through techniques like heatmap visualizations and correlation matrix to reveal complex relationships and patterns.</li>
<li>ANOVA testing: Analysis of variance (ANOVA) is performed to assess whether "drive-wheels" has a significant impact on "price".</li>
<li>Feature selection: Based on the insights gathered during the EDA, important features are selected for model building, and irrelevant or redundant features may be removed to improve model performance.</li>
</ol>
<p>After completing the EDA, a deeper understanding of the dataset is achieved, which will guide the subsequent modeling and evaluation process.</p>
<br><br>



<h4>Model Development:</h4>
<p>In this notebook, the primary focus is on developing various regression models to predict the target variable using the insights gained from the previous exploratory data analysis. The main steps involved in the model development process are:</p>
<ol>
<li>Importing necessary libraries: Required libraries, such as pandas, numpy, scikit-learn, and matplotlib, are imported to facilitate model building, evaluation, and visualization.</li>
<li>Loading the cleaned dataset: The cleaned dataset from the data wrangling step is loaded into a pandas DataFrame for model development and analysis.</li>
<li>Feature selection: Based on the insights gathered during the EDA, important features are selected for model building, focusing on 'horsepower', 'curb-weight', 'engine-size', and 'highway-mpg'.</li>
<li>Linear regression model: A simple linear regression model is built to predict the target variable using a single feature. The model's performance is evaluated using metrics such as R-squared and Mean Squared Error (MSE).</li>
<li>Multiple regression model: A multiple regression model is developed using multiple features to predict the target variable. The model's performance is evaluated using R-squared and MSE.</li>
<li>Polynomial regression model: Polynomial regression models are built to capture the nonlinear relationships between the features and the target variable. Different degrees of polynomial models are explored, and their performance is evaluated using R-squared and MSE.</li>
<li>Model comparison and selection: The performance of the different regression models is compared, and the best model is selected based on its performance metrics and the complexity of the model.</li>

</ol>
<p>After the model development process, the selected model can be fine-tuned, validated, and deployed for making predictions on new data. We conclude that the multiple linear regression model performs better.</p>
<br><br>



<h4>Model Evaluation and Refinement:</h4>
<p>In this notebook, the primary focus is on evaluating and refining the regression models developed in the previous notebook. The main steps involved in the model evaluation and refinement process are:</p>
<ol>
<li>Importing necessary libraries: Required libraries, such as pandas, numpy, scikit-learn, and matplotlib, are imported to facilitate model evaluation, refinement, and visualization.</li>
<li>Loading the cleaned dataset: The cleaned dataset from the data wrangling step is loaded into a pandas DataFrame for further analysis.</li>
<li>Train-test split: The dataset is split into training and testing sets to assess the performance of the developed models on unseen data.</li>
<li>Cross-validation: Cross-validation techniques are employed to obtain a more reliable estimate of the models' performance and to reduce the risk of overfitting.</li>
<li>Grid search: Grid search is used to systematically explore the hyperparameters of various models, such as linear, polynomial, and polynomial with ridge regression, to find the best combination of parameters for each model.</li>
<li>Model comparison and selection: The performance of the different regression models is compared using metrics such as R-squared and Mean Squared Error (MSE), and the best model is selected based on its performance metrics and the complexity of the model.</li>
<li>Model refinement: The selected model is fine-tuned by adjusting its hyperparameters to improve its performance on the testing set.</li>
<li>Model validation: The refined model is validated on the testing set to ensure that it generalizes well to new data and provides reliable predictions.</li>
</ol>
<p>After completing the model evaluation and refinement process, the best model is ready for deployment to make predictions on new data.</p>

