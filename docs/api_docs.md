# PrecisionCustomer API Endpoint Documentation

The **PrecisionCustomer** API is built on FastAPI, offering automatic request validation, OpenAPI specification support, and JSON/Form payload handling.

---

## 🛠️ Endpoints Overview

| Method | Endpoint | Request Content-Type | Response Type | Description |
| :--- | :--- | :--- | :--- | :--- |
| **GET** | `/` | None | HTML | Renders the Glassmorphic web frontend |
| **POST** | `/` | `application/x-www-form-urlencoded` | HTML | Process web form and render prediction results |
| **GET** | `/health` | None | `application/json` | Checks service status |
| **GET** | `/train` | None | `application/json` | Manually triggers the model training pipeline |
| **POST** | `/api/predict` | `application/json` | `application/json` | Segment a single customer profile |
| **GET** | `/api/demo` | None | `application/json` | Returns mock profiles for form prefill |
| **POST** | `/api/predict/file`| `multipart/form-data` | `text/csv` (File) | Batch segment a CSV list |

---

## 🔌 Detail Endpoint Specifications

### 1. Health Status
- **Endpoint**: `/health`
- **Method**: `GET`
- **Response**:
```json
{
  "status": "healthy",
  "service": "precision-customer-segmentation"
}
```

### 2. Segment Single Customer Profile
- **Endpoint**: `/api/predict`
- **Method**: `POST`
- **Request Body (JSON)**:
```json
{
  "Age": 38,
  "Education": 3,
  "Marital_Status": 1,
  "Parental_Status": 1,
  "Children": 1,
  "Income": 62000.0,
  "Total_Spending": 850.0,
  "Days_as_Customer": 300,
  "Recency": 14,
  "Wines": 400,
  "Fruits": 50,
  "Meat": 250,
  "Fish": 30.0,
  "Sweets": 20,
  "Gold": 100.0,
  "Web": 6,
  "Catalog": 3,
  "Store": 5,
  "Discount_Purchases": 2,
  "Total_Promo": 2,
  "NumWebVisitsMonth": 4
}
```
- **Success Response (200 OK)**:
```json
{
  "status": true,
  "prediction": {
    "cluster": 1,
    "persona": {
      "name": "Affluent Shopper",
      "description": "High-income individuals or couples with no children...",
      "spending_tier": "High",
      "strategy": "Target with premium loyalty clubs and exclusive events..."
    }
  }
}
```

### 3. Retrieve Demo Prefill Profiles
- **Endpoint**: `/api/demo`
- **Method**: `GET`
- **Response**: Returns a dictionary mapping cohort labels (`value_shopper`, `premium_aficionado`, `frugal_starter`) to prefilled profiles ready for UI integration.

### 4. Process Batch File (CSV Upload)
- **Endpoint**: `/api/predict/file`
- **Method**: `POST`
- **Request (Multipart Form)**: Form file upload named `file` (must end with `.csv`).
- **Response**: Downloads a CSV file with two appended columns: `predicted_cluster` (int) and `persona_name` (string).
- **Required Columns in CSV**:
  - `Age`, `Education`, `Marital Status` (or `Marital_Status`), `Parental Status`, `Children`, `Income`, `Total_Spending`, `Days_as_Customer`, `Recency`, `Wines`, `Fruits`, `Meat`, `Fish`, `Sweets`, `Gold`, `Web`, `Catalog`, `Store`, `Discount Purchases`, `Total Promo`, `NumWebVisitsMonth`
