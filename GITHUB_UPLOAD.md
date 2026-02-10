# GitHub Upload Instructions

Your local Git repository is ready! Follow these steps to upload to GitHub:

## Step 1: Create GitHub Repository

1. Go to https://github.com/new
2. Enter repository name: `EE5907-Assignment1`
3. Enter description: `Binary classification using MLP and RBF networks - EE5907 Assignment 1`
4. Select **Public** (recommended for portfolio) or **Private**
5. **DO NOT** check "Initialize this repository with a README" (we have one)
6. Click **Create repository**

## Step 2: Copy Your Repository URL

After creating the repo, GitHub shows a page with a URL like:
```
https://github.com/YOUR_USERNAME/EE5907-Assignment1.git
```

## Step 3: Run These Commands

Open PowerShell in the project folder and run:

```powershell
$gitPath = "C:\Program Files\Git\cmd\git.exe"
& $gitPath branch -M main
& $gitPath remote add origin https://github.com/YOUR_USERNAME/EE5907-Assignment1.git
& $gitPath push -u origin main
```

Replace `YOUR_USERNAME` with your actual GitHub username.

## Step 4: Enter GitHub Credentials

When prompted, enter:
- **Username**: Your GitHub username
- **Password**: Your GitHub personal access token (not your password!)

To create a personal access token:
1. Go to https://github.com/settings/tokens
2. Click "Generate new token"
3. Select scopes: `repo` (full control of private repositories)
4. Click "Generate token"
5. Copy the token and paste it when prompted

## Done!

Your repository will be live on GitHub. Share the URL with others or add a link to your portfolio!

---

### Alternative: Use Git Bash

If PowerShell commands give issues, use Git Bash instead:
1. Right-click in folder → "Git Bash Here"
2. Run standard commands:
```bash
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/EE5907-Assignment1.git
git push -u origin main
```
