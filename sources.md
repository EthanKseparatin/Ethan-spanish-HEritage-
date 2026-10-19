# Sources and indicator notes

All historical values are retrieved from the official World Bank Indicators API at runtime. For each indicator, the app requests the period from the current calendar year minus 70 through the current year minus 1. API responses can include missing/null observations, so actual coverage and count are reported rather than assumed.

| Indicator | Code | Units in app | Definition / caution |
|---|---|---|---|
| Life expectancy at birth | `SP.DYN.LE00.IN` | years | Expected years lived by a newborn under current mortality patterns. |
| Unemployment, national estimate | `SL.UEM.TOTL.NE.ZS` | % of labor force | National-estimate unemployment rate. |
| GDP per capita | `NY.GDP.PCAP.CD` | current US$ per person | GDP divided by population; not average wealth or household income. |
| Crude birth rate | `SP.DYN.CBRT.IN` | births per 1,000 people per year | A rate, not a count of births. |
| Net migration | `SM.POP.NETM` | people (net) | Net migration; not an emigration count. |
| Intentional homicide rate | `VC.IHR.PSRC.P5` | per 100,000 people | Intentional homicide victims per 100,000 people. |
| Consumer price inflation | `FP.CPI.TOTL.ZG` | % annual change | Annual percentage change in consumer price index. |
| Central government debt | `GC.DOD.TOTL.GD.ZS` | % of GDP | Central government debt as a share of GDP. |
| Total reserves including gold | `FI.RES.TOTL.CD` | current US$ | Total reserves including gold. |
| External balance on goods and services | `NE.RSB.GNFS.ZS` | % of GDP | External balance on goods and services as a share of GDP. |

## Official links

- World Bank Indicators API documentation: https://datahelpdesk.worldbank.org/knowledgebase/articles/889392-about-the-indicators-api-documentation
- World Bank data catalog: https://data.worldbank.org/
- API pattern: `https://api.worldbank.org/v2/country/{COUNTRY_CODE}/indicator/{INDICATOR_CODE}?format=json&date={START}:{END}`
- Country identifiers are World Bank ISO-style codes: Mexico MEX, Brazil BRA, Argentina ARG, Colombia COL, Chile CHL, Peru PER, Venezuela VEN, Ecuador ECU, Uruguay URY.

## Interpretation and methods

- Brazil is a Latin American comparison but its principal language is Portuguese.
- The app fits a polynomial using centered time `t = year - base year`.
- R², MAE, and residuals describe the fit to the supplied observations. A chronological holdout score is separately labeled.
- The app does not claim that simplified residual bounds are validated 95% prediction intervals.
- The app does not impute missing data or invent historical causes. Any extrapolation beyond observed years should be treated cautiously.
