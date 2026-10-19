# Hispanic Heritage Historical Data Explorer

A self-contained Streamlit classroom app for World Bank historical data, polynomial curve fitting, and an interactive function/calculus lab.

## Run locally

Python 3.10+ is recommended.

```bash
pip install -r requirements.txt
python -m streamlit run app.py
```

## Deploy on GitHub and Streamlit Community Cloud

1. Create a GitHub repository.
2. Upload `app.py` and `requirements.txt` to the repository root.
3. Open Streamlit Community Cloud and connect your GitHub account.
4. Choose the repository and branch, set **Main file path** to `app.py`, then deploy.
5. Review the app logs if the build fails. The app needs outbound HTTPS access to `api.worldbank.org`.

## Deploy on Render

Create a **Python Web Service** connected to the repository.

- Build command: `pip install -r requirements.txt`
- Start command: `streamlit run app.py --server.address=0.0.0.0 --server.port=$PORT`

## App overview

- Fetches official World Bank indicators for the 70 calendar years from current year minus 70 through current year minus 1.
- Shows actual non-null coverage and does not fabricate missing values.
- Keeps source data and editable working data separate; edits are marked as a user-modified scenario.
- Fits a centered-time polynomial and reports in-sample R², MAE, residuals, observation count, and optional chronological holdout MAE.
- Provides single-year and batch estimates, same-indicator country comparisons, contextual function builder, derivatives and numerical calculus analysis, and CSV/XLSX/PDF exports.

## Verification checklist

1. Confirm the indicator code, units, definition, source link, retrieval date, and actual observation count.
2. Change a working-table value and confirm the scenario warning appears; press **Reset edited data to source** and confirm it clears.
3. Try a duplicate year, non-integer year, blank/nonnumeric value, and an indicator constraint violation.
4. Set a polynomial degree higher than the available observation count and confirm the app explains why fitting is unavailable.
5. Compare the displayed equation with centered-time definition `t = year - base year`; check in-sample R², MAE, and residuals.
6. Calculate a single year, then request the same year in batch mode; estimates should agree.
7. Test invalid custom years, reversed year ranges, and years outside the fitted range.
8. Switch country, indicator, degree, sample increment, forward horizon, comparison countries, and printer view; each should change its corresponding output.
9. In the function lab, test `x^3 - 3*x`, `1/(x-2)`, `log(x)`, `exp(0.2*x)`, and `sin(x)`. Use intervals that include excluded domain values and verify warnings/plot gaps.
10. Download all three exports and open the CSV, Excel workbook, and PDF to verify they are genuine formats and match the current selection.
11. Deploy and test again. If API calls fail, check the World Bank API status and the Streamlit/Render logs for timeout, HTTP, or dependency errors.

## Important limitations

This is an educational exploratory model, not a validated forecasting system. Missing values are not imputed. A polynomial can behave implausibly outside observed years. R² and MAE are in-sample metrics unless separately labeled holdout. The function lab uses symbolic differentiation but some root/extrema/domain checks rely on numerical sampling and are approximations. The baseline simulation is frame-slider driven rather than a background timer, to remain stable on hosted Streamlit services.
