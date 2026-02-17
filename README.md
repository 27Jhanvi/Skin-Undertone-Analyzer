💖 Skin Undertone Analyzer
Computer Vision + LAB Color Science + Dual Lighting Stabilization

🔗 Live App: https://skin-undertone-analyzer-miinasr33wkqdaarv7tuyg.streamlit.app/

Problem Statement: Understanding your skin undertone is essential for choosing the right foundation, concealer, jewelry, hair color, and seasonal palette.

However:Most online undertone tools are locked behind paid beauty platforms
Results are often subjective or quiz-based
Many rely on basic image filters rather than real color science
There is limited transparency in how results are calculated

As someone curious about my own undertone and frustrated by paywalled tools ,I decided to build my own data-driven, computer vision-based undertone analyzer.
This project transforms a personal curiosity into a deployable AI product.

Project Goal: To build an accurate, explainable, and lighting-robust skin undertone classification system using:
Computer Vision
HSV-based skin segmentation
LAB color space statistics
Dual lighting stabilization
Streamlit UI deployment

Technical Overview
This project combines image processing and color science techniques:

🔬 1. Skin Pixel Isolation
HSV thresholding for skin region segmentation
Morphological filtering (open + close operations)
Saturation-based mask refinement
Noise reduction pipeline

🎨 2. Color Space Transformation
RGB → LAB conversion
Median a* and b* channel extraction
Robust statistical aggregation across skin pixels

⚖️ 3. Undertone Classification Logic
Warm vs Cool scoring based on LAB a*/b* direction
Neutral band tolerance modeling
Olive detection heuristic
Confidence scoring mechanism
Dual-image averaging for stability

💡 4. Lighting Robustness
Optional Gray-World White Balance correction
Support for:
Natural light image
Indoor light image
Combined prediction for reduced lighting bias

🛠️ Tech Stack
Python
OpenCV
NumPy
scikit-image
Streamlit
Git + GitHub
Streamlit Cloud Deployment

Key Features
✔️ HSV-based dynamic skin masking
✔️ LAB color statistical modeling
✔️ Adjustable neutral tolerance band
✔️ Olive undertone heuristic detection
✔️ Confidence scoring system
✔️ Dual-light image stabilization
✔️ Custom pastel UI with CSS styling
✔️ Fully deployed cloud application

How It Works (User Guide)
1️⃣ Open the Live App link
2️⃣ Upload one image OR
3️⃣ Upload two images (Natural + Indoor for better stability)
4️⃣ Adjust optional sliders if needed:
Saturation refinement
Neutral tolerance band
5️⃣ Click upload
6️⃣ The model analyzes skin pixels and returns:
Your predicted undertone (Warm / Cool / Neutral / Olive)
Confidence score
Optional mask visualization
Debug color statistics (if enabled)
Bingo!!!! you now know your undertone.

Why This Is Different???
Unlike quiz-based beauty tools, this analyzer: Uses real pixel-level color extraction
Operates in LAB perceptual color space
Reduces lighting bias via dual input averaging
Exposes intermediate statistics for transparency
Is fully explainable and reproducible

Engineering Highlights
Built modular pipeline (src/ structured project)
Designed reusable classification logic
Implemented mask visualization overlay
Managed Git version control + merge resolution
Deployed production-ready Streamlit app
Styled custom UI using injected CSS

Future Improvements
Face auto-detection instead of HSV-only segmentation
Adaptive skin-tone clustering
Seasonal color palette recommendation
Makeup / jewelry recommendation engine
Downloadable undertone report
Mobile UI optimization

Author
Built by Jhanvi Varma
Data Science & Applied AI Enthusiast

Final Note
This project started as a personal curiosity 
and evolved into a fully deployed AI-powered beauty tool.
Sometimes the best way to get an answer is to build the system yourself.
