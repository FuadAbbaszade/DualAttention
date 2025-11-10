# GitHub Setup Guide

This guide will help you push your Dual-Attention Whisper project to GitHub.

## Prerequisites

- Git installed on your system
- GitHub account created
- Repository created on GitHub (can be done in step 2)

## Step-by-Step Instructions

### 1. Initialize Git (if not already done)

```bash
cd /DualAttention
git init
```

### 2. Create .gitignore (already exists)

The project already has a `.gitignore` file that excludes:
- Python cache files (`__pycache__`, `*.pyc`)
- Virtual environments (`venv/`, `env/`)
- Model checkpoints (`outputs/`, `*.pt`, `*.bin`)
- Data files (`data/`)
- IDE files (`.vscode/`, `.idea/`)
- Logs and temporary files

### 3. Stage All Files

```bash
# Add all files
git add .

# Verify what will be committed
git status
```

### 4. Create Initial Commit

```bash
git commit -m "Initial commit: Dual-Attention Whisper with HuggingFace integration

- Dual-attention mechanism for noise-robust ASR
- HuggingFace dataset integration
- Optimized training pipeline
- Comprehensive documentation
- GPU memory requirements and recommendations"
```

### 5. Create GitHub Repository

**Option A: Via GitHub Website**
1. Go to https://github.com/new
2. Repository name: `DualAttention` (or your preferred name)
3. Description: "Dual-Attention Whisper for Noise-Robust Speech Recognition"
4. Choose Public or Private
5. **DO NOT** initialize with README (we already have one)
6. Click "Create repository"

**Option B: Via GitHub CLI**
```bash
# Install GitHub CLI if not already installed
# https://cli.github.com/

# Create repository
gh repo create DualAttention --public --source=. --remote=origin
```

### 6. Connect to GitHub Repository

If you created the repo via website:

```bash
# Add remote
git remote add origin https://github.com/YOUR_USERNAME/DualAttention.git

# Verify
git remote -v
```

### 7. Push to GitHub

```bash
# Push to main branch
git branch -M main
git push -u origin main
```

### 8. Verify Upload

Visit your repository URL:
```
https://github.com/YOUR_USERNAME/DualAttention
```

Check that all files are there:
- ✅ README.md displays properly
- ✅ Code files are visible
- ✅ Documentation files are present
- ❌ Large model files are NOT uploaded (they should be in .gitignore)

## Repository Structure on GitHub

```
YOUR_USERNAME/DualAttention/
├── README.md (main project overview)
├── QUICKSTART.md (quick start guide)
├── USAGE_GUIDE.md (detailed usage)
├── CHANGELOG.md (version history)
├── CONTRIBUTING.md (contribution guidelines)
├── LICENSE (MIT license)
├── requirements.txt (dependencies)
├── setup.py (package setup)
├── .gitignore (ignored files)
├── src/ (source code)
│   ├── model/ (dual-attention implementation)
│   ├── data/ (dataset and collators)
│   └── training/ (metrics and utilities)
└── scripts/ (training, inference, evaluation)
```

## Post-Upload Tasks

### 1. Add Topics/Tags

On GitHub repository page:
1. Click the gear icon next to "About"
2. Add topics: `whisper`, `speech-recognition`, `pytorch`, `transformers`, `azerbaijani`, `asr`, `noise-robust`, `dual-attention`
3. Add description: "Dual-Attention Whisper for Noise-Robust Speech Recognition with HuggingFace Integration"
4. Add website (if applicable)
5. Save changes

### 2. Enable GitHub Pages (Optional)

If you want documentation hosted:
1. Go to Settings → Pages
2. Source: Deploy from branch
3. Branch: main, folder: /docs (if you create docs)
4. Save

### 3. Create Release (Optional)

Create a release tag for version 1.0:

```bash
# Create and push tag
git tag -a v1.0.0 -m "Version 1.0.0: Initial release with HuggingFace integration"
git push origin v1.0.0
```

Then on GitHub:
1. Go to "Releases"
2. Click "Create a new release"
3. Select tag: v1.0.0
4. Release title: "v1.0.0 - Initial Release"
5. Description: Copy content from CHANGELOG.md
6. Publish release

### 4. Update README with Repository URL

Add at the top of README.md:

```markdown
# Dual-Attention Whisper for Noise-Robust Speech Recognition

🔗 **Repository**: https://github.com/YOUR_USERNAME/DualAttention
```

Commit and push:
```bash
git add README.md
git commit -m "Add repository URL to README"
git push
```

## Common Issues and Solutions

### Issue: "Repository already exists"
**Solution**: Use a different name or delete the existing repository

### Issue: "Permission denied (publickey)"
**Solution**: Set up SSH keys or use HTTPS with personal access token
```bash
# Generate SSH key
ssh-keygen -t ed25519 -C "your_email@example.com"

# Add to GitHub: Settings → SSH and GPG keys → New SSH key
```

### Issue: "Large files detected"
**Solution**: Remove large files and use Git LFS
```bash
# Install Git LFS
git lfs install

# Track large files
git lfs track "*.bin"
git lfs track "*.pt"

# Add .gitattributes
git add .gitattributes
git commit -m "Configure Git LFS"
```

### Issue: "Files not showing up"
**Solution**: Check .gitignore, ensure files are staged
```bash
# Check what's ignored
git status --ignored

# Force add if needed (be careful!)
git add -f path/to/file
```

## Recommended Workflow

### For Future Updates

```bash
# 1. Make changes
# ... edit files ...

# 2. Stage changes
git add .

# 3. Commit with descriptive message
git commit -m "Add feature X: brief description"

# 4. Push to GitHub
git push
```

### For Collaborators

```bash
# 1. Create feature branch
git checkout -b feature/new-feature

# 2. Make changes and commit
git add .
git commit -m "Implement new feature"

# 3. Push branch
git push -u origin feature/new-feature

# 4. Create Pull Request on GitHub
```

## Next Steps

After pushing to GitHub:

1. ✅ Add comprehensive README badges
2. ✅ Set up GitHub Actions for CI/CD (optional)
3. ✅ Create issues for known bugs or enhancements
4. ✅ Add a LICENSE file (already included)
5. ✅ Share your repository!

## Resources

- [GitHub Docs](https://docs.github.com/)
- [Git Documentation](https://git-scm.com/doc)
- [GitHub CLI](https://cli.github.com/)
- [Markdown Guide](https://www.markdownguide.org/)

---

**Ready to push!** Follow the steps above and your project will be live on GitHub. 🚀
