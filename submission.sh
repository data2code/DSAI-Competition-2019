#!/bin/bash
# Submit a new solution to the AIcrowd leaderboard

# Check for arguments and right context or print usage
if (( $# != 1 )); then
    echo "Usage: submission.sh <solution tag>"
    exit 1
fi

REPO_URL=`git config --get remote.origin.url | awk -F':' '{print $2}' | sed "s/.git$//g"`
SOLUTION_TAG=submission-$1

echo "Submitting solution: $SOLUTION_TAG"

git add run.sh
git add environment.yml
git add aicrowd.json
#git add EDA_simple.py
#git add EDA.py
#git add EDA_v3.py
#git add EDA_v4_fix17-19.py
git add ../E2C.pptx
git add EDA_v4.py
git add configjson.py
git add model.pkl.gz
git add model_lr0.05_nt180_sd90.pkl.gz
git add impute.pkl.gz
git add hyperparameter.best.json
git add hyperparameter.range.json
git add hyperparameter.estimator.json
git add feature_cols.txt
git add submission.sh
git add nltk_data
git add --all countvec
git add --all useful_terms

git commit -m "${SOLUTION_TAG}"
git tag -am "${SOLUTION_TAG}" ${SOLUTION_TAG}
git push origin master
git push origin ${SOLUTION_TAG}

echo "Done"
echo "You can view the created tags in your repository here: https://gitlab.aicrowd.com/${REPO_URL}/tags"
echo "The running status can be seen here: https://gitlab.aicrowd.com/${REPO_URL}/issues"
