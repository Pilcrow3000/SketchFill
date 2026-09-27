# Streamlit Cloud Deployment Guide

## Prerequisites

1. **GitHub account** (free)
2. **Streamlit Cloud account** (free, sign up at share.streamlit.io)
3. **Your trained model checkpoint** (`checkpoints/best_model.pt`)

## Important: Model Size Limitation

⚠️ **Critical Issue**: Your model checkpoint (`best_model.pt`) is ~196MB, but GitHub has a 100MB file size limit, and Streamlit Cloud free tier works best with smaller apps.

### Solutions:

#### Option 1: Use Git LFS (Recommended for GitHub)
```powershell
# Install Git LFS
# Download from: https://git-lfs.github.com/

# Initialize Git LFS
git lfs install

# Track large files
git lfs track "checkpoints/*.pt"
git add .gitattributes
git commit -m "Add Git LFS tracking"

# Add and push model
git add checkpoints/best_model.pt
git commit -m "Add trained model"
git push
```

#### Option 2: Host Model Separately (Recommended for Streamlit)
Upload model to cloud storage and download on startup:

1. Upload `best_model.pt` to Google Drive/Dropbox/Hugging Face
2. Get shareable link
3. App downloads it automatically on first run

**Modified app.py** (already handles missing model gracefully):
```python
@st.cache_resource
def load_resources():
    if not BEST_MODEL_PT.exists():
        # Show error message with instructions
        return None, None, None, None
```

#### Option 3: Demo Mode Without Model
For presentation purposes, you can deploy without the model and show:
- Architecture diagrams
- Pre-generated interpolation images
- Technical documentation

## Deployment Steps

### 1. Prepare Repository

```powershell
# Initialize git (if not already done)
git init
git add .
git commit -m "Initial commit: SketchFill CVAE"

# Create GitHub repository
# Go to github.com/new
# Follow instructions to push

git remote add origin https://github.com/YOUR_USERNAME/SketchFill.git
git branch -M main
git push -u origin main
```

### 2. Deploy to Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click "New app"
3. Select your GitHub repository
4. Set:
   - **Main file path**: `app.py`
   - **Python version**: 3.11
5. Click "Deploy"

### 3. Wait for Build

First deployment takes 5-10 minutes:
- Installs dependencies from requirements.txt
- Sets up environment
- Starts application

### 4. Access Your App

You'll get a URL like:
```
https://YOUR_USERNAME-sketchfill-app-abc123.streamlit.app
```

## File Checklist

Ensure these files are in your repository:

- ✅ `app.py` - Main application
- ✅ `requirements.txt` - Python dependencies
- ✅ `.streamlit/config.toml` - Streamlit configuration
- ✅ `packages.txt` - System dependencies (empty for this project)
- ✅ `models/` - Model architecture files
- ✅ `README.md` - Documentation
- ✅ `.gitignore` - Excludes unnecessary files

## Troubleshooting

### "Model not found" Error

**Expected behavior**: The app shows a clear error message explaining the model needs to be trained locally.

**For demo**: Keep pre-generated images in `checkpoints/*.png` (not gitignored) to show examples even without the model.

### Memory Issues

Streamlit Cloud free tier has 1GB RAM limit. If issues occur:

1. Reduce batch size in generation
2. Limit PCA samples in explorer
3. Consider paid tier for production use

### Slow Loading

First load is slow due to:
- Installing PyTorch (~800MB)
- Loading model (~196MB if included)

**Solution**: Use Streamlit's caching (already implemented with `@st.cache_resource`)

## Local Testing

Before deploying, test locally with production settings:

```powershell
# Install exact versions from requirements.txt
pip install -r requirements.txt

# Run with production config
streamlit run app.py
```

## Alternative: Local Demo

If deployment is challenging, you can demo locally to your teacher:

```powershell
.\run_app.ps1
```

Then show on your laptop screen or share via:
- Screen sharing (Zoom/Teams)
- Local network access (if on same WiFi)
- Record a video demo

## Cost Considerations

**Free Tier** (Sufficient for student project):
- Unlimited public apps
- 1GB RAM per app
- Community support

**Paid Tier** (Not needed):
- More resources
- Priority support
- Custom domains

## Security Notes

- Don't commit sensitive data
- Model checkpoints are public if on GitHub
- Use environment variables for any API keys (none needed for this project)

## Recommended Approach for Student Project

**Best strategy**:

1. **Deploy app WITHOUT model** to Streamlit Cloud
2. **Show interactive UI** and technical documentation
3. **Run model locally** on your laptop for live generation
4. **Use pre-generated images** (already in checkpoints/) to show examples

This avoids the large file limitation while still showing all features!

## Support

- Streamlit Docs: https://docs.streamlit.io
- Community Forum: https://discuss.streamlit.io
- GitHub Issues: For code-related problems

---

**For your presentation**: "I've deployed the application to Streamlit Cloud for easy access. The model runs locally due to size constraints, but the web interface demonstrates the architecture and shows pre-generated examples of interpolation and mixing."
