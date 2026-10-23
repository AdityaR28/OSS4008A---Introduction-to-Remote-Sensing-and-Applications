# OSS4008A: Introduction to Remote Sensing and Applications

Course project repository. Each student has their own folder under `students/` where they add their project data, code and progress notes.

## Repository structure

```
.
├── README.md
├── .gitignore
├── CODEOWNERS            # maps student folders to GitHub usernames (only for students with a username in students.txt)
├── setup_students.sh     # creates a folder per student listed in students.txt
├── students.txt          # one student per line: Last, First [| github_username] (instructor only)
├── students/
│   ├── _template/        # copy of the layout every student folder follows
│   └── <firstname_lastname>/   e.g. hamza_arab
│       ├── README.md     # project description + weekly progress log
│       ├── data/         # small datasets, sample imagery, CSVs
│       ├── code/         # scripts and notebooks
│       └── results/      # figures, maps, outputs
```

## For students

1. Accept the collaborator invite and clone the repo:
   ```
   git clone https://github.com/AdityaR28/OSS4008A---Introduction-to-Remote-Sensing-and-Applications.git
   cd OSS4008A---Introduction-to-Remote-Sensing-and-Applications
   ```
2. Work **only inside your own folder**: `students/<firstname_lastname>/` (for example `students/hamza_arab/`).
3. Create a branch, commit and push:
   ```
   git checkout -b <firstname_lastname>/update
   git add students/<firstname_lastname>/
   git commit -m "Add week 3 NDVI results"
   git push -u origin <firstname_lastname>/update
   ```
4. Open a Pull Request into `main`. It will be reviewed and merged.
5. Update your folder's `README.md` progress log each week.

### Data guidelines

- Keep files under **50 MB**; GitHub rejects files over 100 MB.
- Do **not** commit large raw satellite scenes. Put a download link or script in `data/README.md` instead.
- Use clear names such as `landsat8_ndvi_2026-09.tif`, not `final_v2_new.tif`.
- Never commit passwords, API keys or tokens.

## For the instructor

1. Add each student to `students.txt` as `Last, First` (optionally `Last, First | github_username`).
2. Run `bash setup_students.sh` to create their folders. Re-run it any time you add names or usernames; existing folders are kept.
3. Commit and push.
4. Under **Settings → Collaborators**, invite the students.
5. Optional but recommended: under **Settings → Branches**, protect `main` and require pull request reviews, so students can't overwrite each other's work. Once you have students' GitHub usernames, add them to `students.txt` after a `|`, re-run the script, and enable "Require review from Code Owners" so each student can only change their own folder.
