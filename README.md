FlexDDM 
=======

What is FlexDDM?
----------------

FlexDDM is a Python package that allows for the easy formulation, validation, and testing of new diffusion models with minimal coding. FlexDDM uses a simulation-based approach and includes templates of four leading diffusion models and user-friendly instructions for creating new models in base Python. The package also promotes best practices in model development with automated validation tools for model recovery, parameter recovery, and posterior predictive checks. FlexDDM enhances the accessibility of DDMs, enabling researchers to efficiently develop and test new diffusion models, thereby contributing to new theoretical insights to decision dynamics.

Installation Guide
------------------

#### Installation

Make sure to download Git to your laptop. Here are resources to be able to [download Git](https://git-scm.com/downloads). Use the following git command to download the repository onto your laptop. 

``` shell
git clone https://github.com/joyfan00/FlexDDM.git
```

#### Python
FlexDDM is a Python-based package. It uses Python version 3.12.0. The link to download is [here](https://www.python.org/downloads/release/python-3120/). Download the correct version according to your machine. 

#### Anaconda Environment

FlexDDM uses many built in Python libraries to allow for greater efficiency and accessibility. We recommend creating an Anaconda environment to allow for you to utilize the exact specifications that are necessary for the FlexDDM package. 

To download Anaconda, please utilize the following [link](https://www.anaconda.com/download). 

After you download Anaconda, we have a list of the requirements that are necessary to run FlexDDM in the text file `requirements.txt`. To install them, use the following pip command: 

```shell
pip install -r requirements.txt
```

This will download all necessary libraries to utilize for FlexDDM! Please make sure to go to where your GitHub repository is in the Anaconda terminal. 

How to Fit a Model
------------------

#### Data Format:

The data from a two-choice Flanker task experiment required to fit a model with this package has following structure:

<table style="width:96%;">
<colgroup>
<col width="26%" />
<col width="33%" />
<col width="27%" />
<col width="8%" />
</colgroup>
<thead>
<tr class="header">
<th>Subject ID</th>
<th>Condition</th>
<th>Response Time</th>
<th>Accuracy</th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td>1</td>
<td>1</td>
<td>0.4463</td>
<td>0</td>
</tr>
<tr class="even">
<td>1</td>
<td>0</td>
<td>0.6833</td>
<td>1</td>
</tr>
<tr class="odd">
<td>1</td>
<td>1</td>
<td>0.7243</td>
<td>1</td>
</tr>
<tr class="even">
<td>...</td>
<td>...</td>
<td>...</td>
<td>...</td>
</tr>
</tbody>
</table>

The required columns and names are:

-   Subject-ID: **ID**
-   Condition: **congruent or incongruent trails (for Erikson Flanker task)**
-   Accuracy: **accuracy**
-   Reaction-Time (in seconds): **rt**

FlexDDM comes with an example data-set. It can be accessed anytime:

``` py
FlexDDM/S1FlankerData.csv
```

#### Model Fitting Procedure
```
![Alt text](https://file%2B.vscode-resource.vscode-cdn.net/var/folders/t8/nqklvy6x565bg7rjp1gqrcl40000gn/T/TemporaryItems/NSIRD_screencaptureui_ypXnMX/Screen%20Shot%202024-08-03%20at%2012.14.03%20PM.png?version%3D1722701650821)
```

Model Comparison 
------------------
``` py
# Fitting multiple models to the example input data
dstp = DSTP()
ssp = SSP()
dmc = DMC()

runsimulations.run_simulations([dstp, ssp, dmc], 1, 50, 'S1FlankerData.csv', return_dataframes=True)

```

The function `runsimulations.run_simulations` above will use the package's example input data, 'FlexDDM/S1FlankerData.csv', to fit 3 models: the Dual-Stage Two Phase model, the Shrinking Spotlight model, and the Diffusion Model for Conflict. FlexDDM features a total of **4 models** and **custom models can be easily created and included**. See section "Create New Theoretical Models" for more details.

For each model being fit, the `run_simulations` function saves a \*csv file with the extracted parameters for each subject. Furthermore, two fitting statistics, the **Likelihood Chi-Squared (G<sup>2</sup>) statistic** and the **Approximate Bayesian Information Criterion (aBIC)** are used to measure how closely the model fits each participant. 

The first statistic, **Likelihood Chi-Squared (G<sup>2</sup>)** groups empirical and simulated response times by trial type (congruent vs incongruent) and accuracy (correct vs incorrect) and compares the proportions falling within empirical percentiles. A **G<sup>2</sup>** value of 0 indicates that the model is a perfect fit to the participant's data, while a value closer to 1 indicates a poor fit. 

The **approximate Bayesian information criterion (aBIC)** statistic also measures a model's fit while applying a penalty to models with a greater number of parameters. A smaller **aBIC** value indicates better fit. 

`runsimulations.run_simulations` can intake a list of models to allow for easy model comparison. Using the **G<sup>2</sup>** and **aBIC** values, the user can clearly determine which model best describes their participants' latent decision making processes.


Validation Tools
------------------
FlexDDM also includes a suite of tools to validate each model created and to ensure that discovered parameters are recoverable. Because FlexDDM allows for the easy creation of new theoretical models, it's critical to provide user-friendly model and parameter recovery that enforce standard modeling processes. Through using these validation tools, the FlexDDM user can discern whether their new theoretical models uniquely capture a distinct decision-making behavior or if the behavior is better explained by an existing model. 

### Model Validation

The `model_recovery` function is included in the `validationtools.py` file and is essential for validating new theoretical models and ensuring they capture the behavior they intend to. Model recovery assesses a model's accuracy by seeing how well a model can accurately recover or fit to data it generates.

#### How It Works

1. **Simulation**: For each model, the function generates synthetic data using randomly initialized parameters within the model's bounds. This is repeated for a specified number of simulations (default 50).

2. **Model Fitting**: The simulated data from each model is then fit to all implemented models in the package.

3. **Model Comparison**: The Bayesian Information Criterion (BIC) is used to compare the fit of each model to the simulated data. The model with the lowest BIC is considered the best fit for that particular dataset.

4. **Recovery Analysis**: The function keeps track of which model was best able to fit the data generated by each model. This information is compiled into a confusion matrix.

5. **Visualization**: The results are visualized as a heatmap, where each cell represents the probability of a particular model (row) being identified as the best-fitting model for data generated by another model (column).

#### Interpretation

- The diagonal of the heatmap represents how often each model correctly recovers its own data.
- Off-diagonal elements show how often a model is mistakenly identified as the best-fitting model for data generated by another model.
- Ideally, we want to see high probabilities along the diagonal and low probabilities elsewhere, indicating that each model can reliably recover its own data and is distinguishable from other models.

By including this functionality, our package allows researchers to validate their models and ensure that they are capturing the intended behavioral patterns, providing confidence in the model selection process and the interpretability of results when applied to real data.
### Parameter Recovery

The `param_recovery` function in the `validationtools.py` file  is a critical validation tool that assesses the reliability of a model's parameter estimation. This function allows researchers to understand the reliability of their parameter estimates and ensure that the inferences drawn from fitted models are trustworthy and meaningful in the context of their research questions.

#### How It Works:

1. **Data Simulation**: For each model, parameter values within the model's specified bounds are generated and used to simulate response time data. This process repeats for the specified number of trials.

3. **Model Fitting**: The original model is then fit back to the data it simulated to estimate the underlying parameters.

4. **Parameter Comparison**: The function compares the originally generated parameters to the estimated parameters from the model fitting process.

5. **Correlation Analysis**: For each parameter, the function calculates the Pearson correlation coefficient between the generated and recovered parameter values.

6. **Visualization**: The results are visualized as scatter plots for each parameter, showing the relationship between the simulated and fitted parameter values.

#### Interpretation:

- Each scatter plot represents one parameter, with simulated values on the x-axis and fitted values on the y-axis.
- A strong positive correlation (close to 1) indicates good parameter recovery.
- The closer the points are to the diagonal line, the better the recovery.
- Correlation values and their significance are reported for each parameter.

### Posterior Predictive Checks

The posterior predictive check is a built-in validation tool within our fitting function. It's designed to assess how well a fitted model can reproduce the observed data, providing a visual comparison between the model's predictions and the actual experimental results.

#### How It Works

1. **Model Fitting**: The function fits the specified model(s) to the input data for each participant.

2. **Data Simulation**: Using the fitted parameters, the function simulates new data for each participant.

3. **Data Preparation**: Both the experimental and simulated data are organized into a DataFrame, including reaction times, congruency, and accuracy information.

4. **Visualization**: The function creates a 2x2 grid of kernel density estimation (KDE) plots, representing:
   - Congruent, Correct responses
   - Congruent, Incorrect responses
   - Incongruent, Correct responses
   - Incongruent, Incorrect responses

5. **Comparison**: Each plot overlays the distribution of simulated reaction times with the distribution of experimental reaction times.

#### Interpretation

- **Distribution Overlap**: The degree of overlap between the simulated and experimental distributions indicates how well the model captures the observed data patterns.
- **Condition-Specific Performance**: By separating the plots by congruency and accuracy, we can assess the model's performance across different experimental conditions.
- **Sample Size**: Each plot is annotated with the number of data points (N) for that specific condition, providing context for the reliability of the comparison.


Creating New Theoretical Models
------------------
Theoretical models can also be created if they have the same form has the included models in our package. Four models, dual-stage two-phase (DSTP), diffusion model for conflict (DMC), shrinking spotlight (SSP), and the standard drift-diffusion model code have been provided for reference in the `FlexDDM/models` folder under their respective names.

The basis of FlexDDM are classes. Classes can create what are called objects that can have various functionalities and attributes. For example, let's say we have a class Dog. A functionality of the dog is that it can bark, and attributes (or variables) of the dog would be the breed of dog, the name of the dog, and the color of the dog. An example of a dog class would look like this: 

```py
class Dog:
    global name
    global breed 
    global color
    def __init__ (name, breed, color):
        self.name = name
        self.breed = breed
        self.color = color
    def bark():
        print('woof!')
```

The `__init__` allows for users to create the object with certain characteristics, essentially "initialize" the object. In our case, we want the user to define the three attributes, the name, the breed, and the color. When we create an object of a dog, this is what we would do: 

```py
d = Dog('Macey', 'Golden Doodle', 'golden') # creates the object with the dog's name being Macey, the breed being Golden Doodle, and the color being golden, goes in the order of the attributes listed
d.bark() # this will print out woof to the screen
```

Now, let's get started with creating theoretical models for FlexDDM! The first step is to create a new Python file under the `flexddm/models` folder. Label the file the same name as your model. For our example, let's name the file `NewModel.py` because we will create a model called `NewModel`. 

In our `NewModel.py` file, we want to import different libraries. Here is the list that you will need for our example: 
```py
import numpy as np
import numba as nb
from .Model import Model
from flexddm import _utilities as util
```

The `numpy` library is a scientific computing library that allows for efficient calculation of mathematical and statistical operations. The `numba` library allows us to more efficiently run code. Python is what we call an interpreted language, which means that the computer has to first interpret the code and then run the code, which can be a tedious process. However, there are what are called compiled languages, which the computer can run directly, which makes the code run faster. `numba` allows the Python code to be converted from an interpreted language to a compiled language. Anytime you also see the `import _____ as __`, when utilize the library's function in your code, make sure to use the abbreviated version for ease of use (i.e. use np for numpy and nb for numba). The line `from .Model import Model` imports the base Model class that we will talk about in the next paragraph. 

FlexDDM utilizes polymorphism, which means that there are parent and child classes and the child classes inherit the functionality of the parent classes. Whenever you create a new model, you want to create a new class that inherits the parent class Model. This parent Model class includes our customized fitting procedure. The first step is to create the class, which is the following code: 

```py
class NewModel (Model):
```
Afterwards, we define global variables, which are places to store information that are utilized within the entire class. We have the following 8 variables: data, param_number, bounds, parameter_names, DT, VAR, NTRIALS, and NOISESEED. Here is a brief description of each: 
* **data**: contains the data for the model wanted to fit the model to, this will be defined when creating a NewModel object
* **param_number**: indicates the number of parameters
* **bounds**: the bound values of each of the parameters
* **parameter_names**: a list of the parameter names
* **DT**: the change in time to simulated data for 
* **VAR**: standard deviation of the diffusion process
* **NTRIALS**: the number of trials for simulated data 
* **NOISESEED**: the seed of randomization to help with data selection 

This is an example of how they could be defined. Please note that parameter_names and param_number should match. 
```py
global data # keep this as is 
global bounds # keep this as is
parameter_names = ['alpha', 'beta', 'delta', 'sigma', 'tau'] # this is a list, feel free to change the names of the items in the list
param_number = len(self.parameter_names) 
DT = 0.001 # keep this as is
VAR = 0.1 # keep this as is
NTRIALS = 100 # feel free to customize this number 
NOISESEED = 50 # feel free to customize this number
```

In `param_number`, `len` is a function that allows us to retrieved the length of the list and how many parameters are in the parameter_names list. 

Next, let's create our constructor, our way to initialize the `NewModel` object. This is defined by `__init__`. `def` means the creation of a function, which is a way to call different pieces of code. 

```py
def __init__(self, data=None):
    self.modelsimulationfunction = NewModel.model_simulation
    if data != None:
        if isinstance(data, str): 
            self.data = self.getRTData(data)
        else:
            self.data = data
        self.bounds = [(0,20),(0,20),(0,1),(-10,10),(-10,10),(0,min(self.data['rt']))] # these are the bounds for model fitting
    else: 
        self.bounds = [(0,20),(0,20),(0,1),(-10,10),(-10,10),(0,5)] # these are the bounds for model validation 
    super().__init__(self.param_number, self.bounds, self.parameter_names)
```

If you notice, the first line creates a new variable to reference the current class's model simulation function. Make sure to replace `NewModel` with the name of your current class name. The bounds for model fitting and validation are slightly different. Every parameter is the same except for the `tau` parameter. For model validation, we want to give a wider range, so we set that to be (0,5). For model fitting, we want it to be set to (0, min(self.data['rt'])) as we want the bounds to be constrained to the minimum possible response time. 

Now, we will move on to the model simulation function. The model simulation function uses `numba.jit()`, which allows the Python code to run faster. The reason is because Python is what we call an interpreted language, which means that it is read and executed by some other program, which means it takes longer to run. However, `numba.jit()` converts the code to be compiled, which means that it will run faster because it directly executes on your machine. While this allows for greater efficiency, one thing to note is that the any mutable object or returning object must be used from the numpy library. What does this mean? Any data structure (or a way to store information) that allows for the user to contain multiple values of information, must be from the numpy library. 

Let's use a standard drift diffusion model simulation code as a basis: 

```py
@nb.jit(nopython=True, cache=True, parallel=False, fastmath=True, nogil=True)
def model_simulation (alpha_c, alpha_i, beta, delta_c, delta_i, tau, dt=DT, var=VAR, nTrials=NTRIALS, noiseseed=NOISESEED):
    choicelist = [np.nan]*nTrials
    rtlist = [np.nan]*nTrials
    congruencylist = ['congruent']*int(nTrials//2) + ['incongruent']*int(nTrials//2) 
    np.random.seed(noiseseed)
    updates = np.random.normal(loc=0, scale=var, size=10000)
    for n in np.arange(0, nTrials):
        if congruencylist[n] == 'congruent':
            alpha = alpha_c
            delta = delta_c
        else:
            alpha = alpha_i
            delta = delta_i
        t = tau # start the accumulation process at non-decision time tau
        evidence = beta*alpha/2 - (1-beta)*alpha/2 # start our evidence at initial-bias beta
        np.random.seed(n)
        while evidence < alpha and evidence > -alpha: # keep accumulating evidence until you reach a threshold
            evidence += delta*dt + np.random.choice(updates) # add one of the many possible updates to evidence
            t += dt # increment time by the unit dt
        if evidence > alpha:
            choicelist[n] = 1 # choose the upper threshold action
        else:
            choicelist[n] = 0  # choose the lower threshold action
        rtlist[n] = t
    return (np.arange(1, nTrials+1), choicelist, rtlist, congruencylist)
```

The first line 
```py
@nb.jit(nopython=True, cache=True, parallel=False, fastmath=True, nogil=True)
```
uses the `numba` compiler to convert the Python code to compiled language code to make the code run faster. This is because the model simulation code can be extremely time inefficient as it runs for every participant and every time to fit the model slightly better. 

Then, we define the model_simulation function: 
```py
def model_simulation (alpha_c, alpha_i, beta, delta_c, delta_i, tau, dt=DT, var=VAR, nTrials=NTRIALS, noiseseed=NOISESEED):
```
We create the `model_simulation` function using the key word `def` and then in parantheses put in any variables necesssary. `alpha_c`, `alpha_i`, `beta`, `delta_c`, `delta_i`, and `tau` are all the parameters for the model. The other parameters-- `dt`, `var`, `nTrials`, and `noiseseed`-- are pre-standardized parameters that help with simulating participant data. Now, we define some key variables that are used throughout the model simulation function: 
```py
choicelist = [np.nan]*nTrials
rtlist = [np.nan]*nTrials
```
The `choicelist` keeps track of the different choice decisions that the participant makes for the simulated data. The `rtlist` keeps track of the reaction time for a participant to make a decision for the simulated data. Both of these are initially defined to be 
```py 
[np.nan]*nTrials
``` 
which make an array of length `nTrials` and initializes each item within the list to be `np.nan`, which means not a number. `np.nan` is a placeholder for future values that will be added to these arrays. The reason why we have to make these arrays is because of `numba`, we cannot have mutable objects like lists because of `numba`. 

Another key variable we initialize is our congruencylist. 
```py
congruencylist = ['congruent']*int(nTrials//2) + ['incongruent']*int(nTrials//2) 
```
Due to the fact that we assume that there are equal congruent and incongruent trials, we just divide incongruent and congruent trials. Lastly, we set a seed for reproducibility of results and sample 10000 points of normal distribution. We need the seed because it helps to make sure the fitting process is happening accurately and does not introduce any randomness between iterations in the optimization algorithm. 
```py
np.random.seed(noiseseed)
updates = np.random.normal(loc=0, scale=var, size=10000)
```
Then, we iterate through each of the trials to simulate a choice and response time for that trial. 
```py
for n in np.arange(0, nTrials):
```
In each trial, we execute the following process. 

We check the congruency of the trial because we want to utilize a different alpha and delta depending on the congruency. In this case, we set the `alpha` for congruent trials to be `alpha_c`, `delta` for congruent trials to be `delta_c`, `alpha` for incongruent trials to be `alpha_i`, and `delta` for incongruent trials to be `delta_i`. 
```py
if congruencylist[n] == 'congruent':
   alpha = alpha_c
   delta = delta_c
else:
   alpha = alpha_i
   delta = delta_i
```

Before diffusion begins, we want to define the time it starts and relative evidence for each choice. We initialize starting time `t` at `tau`, which represents the non-decision time. We want to account for the total amount of time for this entire process including non-decision time, not just the time that diffusion takes. Also, we initialize `evidence` to be `beta * alpha/2 - (1-beta) * alpha/2`. It determines the initial evidence in a decision-diffusion model by adjusting for starting point bias. It does this by taking half of the boundary separation (α) and weighting it based on the starting point bias (β) and its complement (1-β). The result represents how the starting point bias affects the initial evidence in favor of one decision over the other.
```py
t = tau # start the accumulation process at non-decision time tau
evidence = beta * alpha/2 - (1 - beta) * alpha/2 # start our evidence at initial-bias beta
```

To reproduce the results for each particular trial, we utilize a different seed for each trial to sample a unique set of values for updates but consistently across iterations of the optimization algorithm. 
```py
np.random.seed(n)
```

We then accumulate evidence by the average rate `delta` scaled by the time increment `dt` and adding some noise sampled from `updates`. We also increase time by the time increment `dt`. This process continues until evidence passes one of the thresholds. 
```py
while evidence < alpha/2 and evidence > -alpha/2: # keep accumulating evidence until you reach a threshold
      evidence += delta*dt + np.random.choice(updates) # add one of the many possible updates to evidence
      t += dt # increment time by the unit dt
```

Once the evidence accumulation process completes and reaches a threshold, we then check which threshold was crossed. If it crosses the higher threshold, we record 1 in `choicelist` at the specific trial number index, otherwise we record 0. Additionally, we track the current time `t` in `rtlist` at the specific trial number index. 
```py
if evidence > alpha/2:
   choicelist[n] = 1 # choose the upper threshold action
else:
   choicelist[n] = 0  # choose the lower threshold action
rtlist[n] = t
```

That concludes the process in the for loop that iterates through each trial. Once the evidence accumulation process for each trial has been complete, we now return the trial number, choice, RT, and congruency for each simulated trial. 
```py
return (np.arange(1, nTrials+1), choicelist, rtlist, congruencylist)
```


References
---
- LaFollette, K., Fan, J., Puccio, A., & Demaree, H. A. (2024). FlexDDM: A flexible decision-diffusion Python package for the behavioral sciences. *Proceedings of the Annual Meeting of the Cognitive Science Society, 46*. Retrieved from [https://escholarship.org/uc/item/4q57r2x0](https://escholarship.org/uc/item/4q57r2x0)

- LaFollette, K. J., Fan, J., Puccio, A., & Demaree, H. A. (Under Review). Democratizing diffusion decision models: A comprehensive tutorial on developing, validating, and fitting diffusion decision models in Python with FlexDDM. Retrieved from [https://doi.org/10.31234/osf.io/j9m67](https://doi.org/10.31234/osf.io/j9m67)

- Ratcliff, R., & Smith, P. L. (2004). A comparison of sequential sampling models for two-choice reaction time. *Psychological Review*.
- Ratcliff, R., & McKoon, G. (2016). Decision Diffusion Model: Current Issues and History. *Trends in Cognitive Science*.
- White, C. N., Ratcliff, R., & Vasey, M. W. (2017). Testing the validity of conflict drift-diffusion models for use in estimating cognitive processes: A parameter-recovery study. *Psychonomic Bulletin & Review*.
- Ahmad, I., Hamid, M., & Bakar, M. S. (2022). Differential Evolution: A Recent Review Based on State-of-the-Art Works. *Applied Soft Computing*.
- Wang, Q., & Shoup, T. (2011). Parameter sensitivity study of the Nelder-Mead Simplex Method. *Advances in Engineering Software*.
