# Audit and release evidence

## Repository and source

GitHub main was `48275e85a970f1dfb8b5111ced48b064f4f51b56` at the start of review. It contains the current FastAPI app, reusable feature transformer, evaluation notebook and two tracked copies of the older project. The outer local folder is a source export. Its application, research and README match the canonical GitHub tree apart from line endings. The two local Git checkouts pointed to the older commit `91949977035b747c5470f835d4cf93cb7a54190c`. The first checkout reported the second checkout as untracked. Both local checkouts and the local SQLite file were preserved. The release branch is based on a fresh clone of current main.

The duplicate trees match each other and preserve an older FastAPI app and notebook in existing Git history. The current tree removes both copies, the tracked bytecode, the obsolete HTML helper, the redundant notebook model, and the database placeholder. Git history retains files published by previous revisions.

## Data and model provenance

The Kaggle dataset API identified the source as **Titanic Dataset** by **M Yasser H**, version 1, listed as CC0. The public archive's CSV and the local source file both hash to `7d118fef8b6ccf7f81111877bc388536f7b1e498a655e3d649d19aaa010e9f6f`. The 891 labeled records and columns match the format of the Kaggle Titanic competition training set. The mirror listing documents its declared license. Its listing does not establish a separate upstream chain of title.

- [Dataset and license listing](https://www.kaggle.com/datasets/yasserh/titanic-dataset)
- [Competition data documentation](https://www.kaggle.com/competitions/titanic/data)

The raw CSV stays local and is excluded from the deployment. The original notebook displayed identifiable passenger rows in its outputs. The revised notebook reports aggregate analyses and is committed without execution outputs. The original app and notebook artifacts had the same SHA-256 and referenced scikit-learn 1.8.0 in their serialization metadata. The baseline artifact was inspected as bytes. The release model is generated from the checked-in trainer with a checksum and package-version manifest.

## Evaluation protocol

The original notebook evaluated the dummy baseline on the test partition before finishing model selection. It also plotted target relationships across the full table before splitting and inspected identifiable misclassifications. Preprocessing was already inside a scikit-learn pipeline and cross-validation fit clones on training folds. The baseline's test report and the full-data exploration created test-set exposure without establishing that a particular choice was changed by it.

An isolated Python 3.12 environment executed all 19 original notebook code cells with Jupyter. The baseline test accuracy was 0.614525. The original and revised final evaluations both returned accuracy 0.826816, precision 0.787879, recall 0.753623, F1 0.770370 and ROC AUC 0.865349. The confusion matrix remained 96 true negatives, 14 false positives, 17 false negatives and 52 true positives.

The revised protocol uses training-only exploration, stratified cross-validation and baseline comparison. It chooses the classifier before test scoring and retains seed 42 for comparison with the original report. The test partition has been inspected before. This is a retrospective repeat evaluation. Family or ticket groups may appear in both partitions, and the single historical event limits generalization.

## Serving and deployment

The previous application stored submitted details in SQLite and exposed shared history and deletion. The updated flow uses same-origin JSON inference with explicit titles and a cabin-recorded flag. It accepts no names or free-text fields. Input validation, request-size limits, safe error responses and response headers apply at the API boundary. The app serves only its HTML and static asset directory. It verifies the packaged model against its digest and runtime dependencies before loading it from a fixed server-owned location.

The Vercel deployment uses Python 3.12 and FastAPI. The release gate includes local checks, a reviewed PR with successful CI, deployment readiness and live anonymous smoke checks. Production data dependencies consist of the fitted pipeline and its evaluation manifest.
