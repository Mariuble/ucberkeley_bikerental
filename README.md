# Bike Sharing in Washington D.C. Dataset
UC Berkeley: INFO 251 final project:  
Authors:
- Marius Lerstein
- Tuva Cornelia Oppenhagen

# Running the code
There are two main methods to run the code:  

(1) You can either go through the notebooks and run through each model.  
- Make sure to set up your environment before running the notebooks, I have not provided an environment or container for this.

(2) You can run the ./train_and_test.sh script in the /scripts directory.
- This approach only implements the fine-tuned gradient boosting regressor algorithm to avoid rewriting the whole notebook again.

For the second method:
1. Make sure you have the required packages installed. I have created requirements.txt file which you can build your virtual environment on.
2. `cd` into the script directory from your terminal.
3. Run `./train_and_test.sh`.
4. Look at the printed R-squared value to evaluate model.
