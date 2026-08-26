# AI Risk Manager: Credit Card Fraud Detection

**Production-grade fraud detection system for Indian BFSI** targeting credit card and transaction fraud with measured precision and recall.

##  Overview

Stop merchants losing money to fraud with an AI-powered fraud detector built on real Indian transaction patterns.

### Key Features
- ✅ **54.21% Fraud Recall** - Catches majority of fraudulent transactions
- ✅ **₹2.61 Crores Net Business Value** - Quantifies fraud prevented vs false positive costs
- ✅ **Indian BFSI Context** - Realistic merchants, amounts in ₹, IST timezone
- ✅ **Production Ready** - Flask API with 4 endpoints, Docker deployment
- ✅ **Transparent Evaluation** - Full clarity on precision/recall trade-offs

##  Model Performance

| Metric | Value |
|--------|-------|
| **Precision** | 10.82% |
| **Recall** | 54.21% |
| **F1-Score** | 0.1804 |
| **ROC-AUC** | 0.6920 |

### Business Impact
- **Fraud Caught**: 2,732 transactions
- **Fraud Value Prevented**: ₹2,73,20,000
- **False Positive Cost**: ₹11,25,750
- **Net Business Value**: ₹2,61,94,250 

##  Quick Start

```bash
pip install -r deployment/requirements.txt
python src/api.py
```

##  API Endpoints

- `GET /health` - Health check
- `GET /info` - Model metadata
- `POST /predict` - Single/batch prediction
- `POST /batch-predict` - Batch with statistics


##  Model Details

- **Algorithm**: XGBoost
- **Features**: 42 (PCA + engineered)
- **Training**: 70-30 stratified split

## Presentations

 **[AI Risk Manager Pitch Deck (View Online)](https://vitacin-my.sharepoint.com/:p:/g/personal/abhisha_chakrabarti2022_vitstudent_ac_in/IQBrB_zVvxNTSJd2tC_T6JTGAYcTIozuqZ031uS_H_GQXjE?e=7EuK9e)**
- 11-slide overview for recruiters and investors
- View online or download

**Contents:**
- Credit card fraud problem (₹2,000+ Cr annually)
- AI Risk Manager solution (42 features, XGBoost)
- Business results (₹2.61 Cr profit)
- Technical architecture
- 2 AM debugging story
- Roadmap for improvements


