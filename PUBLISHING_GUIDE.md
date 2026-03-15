# How to Publish mleval — GitHub + PyPI

---

## Part 1 — GitHub

### Step 1: Create the repo
1. Go to https://github.com/new
2. Repository name: `ml-evaluator`
3. Description: `Model Evaluation Toolkit — bias-variance, ROC, model summary & multi-model comparison`
4. Set to **Public**
5. Do NOT add README or .gitignore (we already have them)
6. Click **Create repository**

### Step 2: Push the code
Open a terminal inside the `mleval/` folder and run:

```bash
git init
git add .
git commit -m "Initial release v1.0.0"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/mleval.git
git push -u origin main
```

Replace `YOUR_USERNAME` with your GitHub username.

### Step 3: Create a Release tag
```bash
git tag v1.0.0
git push origin v1.0.0
```

Then on GitHub:
- Go to your repo → **Releases** → **Create a new release**
- Tag: `v1.0.0`
- Title: `mleval v1.0.0 — Initial Release`
- Paste the contents of CHANGELOG.md in the description
- Click **Publish release**

---

## Part 2 — PyPI

### Step 1: Create accounts
- **PyPI** (production): https://pypi.org/account/register/
- **TestPyPI** (for testing): https://test.pypi.org/account/register/

Enable 2FA on both — PyPI requires it.

### Step 2: Create API tokens
On PyPI:
1. Go to **Account Settings** → **API tokens**
2. Click **Add API token**
3. Name: `mleval-publish`
4. Scope: **Entire account** (first time) or the project after first upload
5. Copy the token — you only see it once!

### Step 3: Install publishing tools
```bash
pip install build twine
```

### Step 4: Test on TestPyPI first
```bash
# Inside the mleval/ folder:

# Build the distribution files
python -m build

# This creates:
#   dist/mleval-1.0.0.tar.gz
#   dist/mleval-1.0.0-py3-none-any.whl

# Upload to TestPyPI
twine upload --repository testpypi dist/*
# Username: __token__
# Password: paste your TestPyPI API token

# Test the install
pip install --index-url https://test.pypi.org/simple/ mleval
python -c "import mleval; print(mleval.__version__)"
```

### Step 5: Publish to PyPI
```bash
twine upload dist/*
# Username: __token__
# Password: paste your PyPI API token
```

That is it. Now anyone can run:
```bash
pip install mleval
```

---

## Part 3 — Update pyproject.toml

Before publishing, open `pyproject.toml` and replace `YOUR_USERNAME` with your real GitHub username in the `[project.urls]` section:

```toml
[project.urls]
Homepage   = "https://github.com/YOUR_USERNAME/mleval"
Repository = "https://github.com/YOUR_USERNAME/mleval"
Issues     = "https://github.com/YOUR_USERNAME/mleval/issues"
```

Also update the `authors` field:
```toml
authors = [
  { name = "Your Name", email = "your@email.com" }
]
```

---

## Part 4 — How to release a new version

When you add features or fix bugs:

1. Update the version in `pyproject.toml` and `mleval/__init__.py`:
   ```
   version = "1.1.0"
   __version__ = "1.1.0"
   ```

2. Add an entry to `CHANGELOG.md`

3. Commit and tag:
   ```bash
   git add .
   git commit -m "Release v1.1.0"
   git tag v1.1.0
   git push origin main --tags
   ```

4. Rebuild and upload:
   ```bash
   rm -rf dist/
   python -m build
   twine upload dist/*
   ```

---

## Quick reference — all commands

```bash
# First-time setup
pip install build twine

# Build
python -m build

# Test upload
twine upload --repository testpypi dist/*

# Production upload
twine upload dist/*

# Check what's on PyPI
pip install mleval
python -c "import mleval; print(mleval.__version__)"
```

---

## After publishing — useful links

| Link | What it is |
|------|-----------|
| `https://pypi.org/project/mleval/` | Your package page on PyPI |
| `https://github.com/YOUR_USERNAME/mleval` | Source code |
| `pip install mleval` | How anyone installs it |
| `pip install git+https://github.com/YOUR_USERNAME/mleval` | Install directly from GitHub |
