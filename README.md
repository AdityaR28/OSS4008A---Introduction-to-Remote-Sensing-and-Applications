# OSS4008A - Introduction to Remote Sensing and Applications

Update this repository regularly to keep track of your progress. It will also be used for your evaluation later on.

## Key points from the first progress meeting

1. All coding has to be done in **Python**.
2. Everything has to be done strictly on **Google Colab**, to avoid library or dependency issues and so that your project can be reproduced by anyone.
3. Work only inside your own folder, `students/firstname_lastname/` (for example `students/hamza_arab/`).
4. Update the progress log in your folder's `README.md` every week.

## Your folder

```
students/firstname_lastname/
├── README.md    project description and weekly progress log
├── data/        small datasets, sample imagery, CSVs
├── code/        Colab notebooks (.ipynb) and Python scripts
└── results/     figures, maps, outputs
```

## Using Colab to pull and push

### One-time setup: create a GitHub token

Colab needs a token instead of your password to push to GitHub.

1. On GitHub, go to your profile picture, then **Settings**, then **Developer settings**.
2. Open **Personal access tokens**, then **Tokens (classic)**, then **Generate new token (classic)**.
3. Name it `colab-oss4008a`, set an expiry that covers the course, and tick the **repo** box.
4. Click **Generate token** and copy it straight away. GitHub only shows it once.
5. You must have accepted the collaborator invite for this repo first (check your email or the Invitations page on GitHub).

Never type the token directly into a notebook cell, because it would be saved in the notebook. The code below asks for it with a hidden prompt.

### Step 1: Clone the repo (every time you open a new Colab session)

Colab sessions start empty, so run this first. Fill in the first four lines.

```python
import os
from getpass import getpass

GH_USER  = "your-github-username"
GH_EMAIL = "your-email@example.com"
FOLDER   = "firstname_lastname"   # your folder name inside students/
OWNER    = "AdityaR28"
REPO     = "OSS4008A---Introduction-to-Remote-Sensing-and-Applications"

token = getpass("GitHub token: ")

os.chdir("/content")
if not os.path.exists(f"/content/{REPO}"):
    !git clone https://{GH_USER}:{token}@github.com/{OWNER}/{REPO}.git
os.chdir(f"/content/{REPO}")

!git config user.name "{GH_USER}"
!git config user.email "{GH_EMAIL}"
!git pull
```

### Step 2: Work in your folder

Save everything into your own folder so it can be pushed, for example:

```python
df.to_csv(f"students/{FOLDER}/data/my_data.csv", index=False)
fig.savefig(f"students/{FOLDER}/results/ndvi_map.png", dpi=150)
```

To save your Colab notebook itself, use **File > Save a copy in GitHub**, choose this repo, and set the path to `students/firstname_lastname/code/your_notebook.ipynb`. You can also use **File > Download > .ipynb** and upload it through the GitHub website.

### Step 3: Push your work

```python
!git add students/{FOLDER}
!git commit -m "Week 3: added NDVI results"
!git pull --rebase
!git push
```

Use a short message that says what you did. Push at least once a week, even if the work is not finished.

### Pulling the latest changes

```python
!git pull
```

### Common problems

- **Authentication failed:** the token is wrong, expired, or missing the `repo` box. Create a new one.
- **Repository not found:** you have not accepted the collaborator invite yet.
- **Push rejected:** run `!git pull --rebase` and then `!git push` again.
- **File too large:** GitHub rejects files over 100 MB. See the data guidelines below.

## Data guidelines

- Keep each file under **50 MB**.
- Do not commit large raw satellite scenes. Put a download link or the download code in `data/README.md` instead.
- Use clear file names such as `landsat8_ndvi_2026-09.tif`, not `final_v2_new.tif`.
- Never commit passwords, API keys or tokens.
