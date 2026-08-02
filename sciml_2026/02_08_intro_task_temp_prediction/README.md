# Petnica Temperature Prediction
### *Can you guess how hot it's going to be? 🌄*

Now that we all have settled, nicely tucked in the hills near Valjevo, it's a good place to notice that weather is a **genuinely** hard prediction problem. One day the valley is baking, the next there's a cold front rolling in off nowhere. 

This is meant as a warm-up to see whether you can fit some simple models that you have heard about yesterday and today. You've already seen the core ideas in your ML lectures, e.g. regression, train/test splits, evaluation metrics. Here you get to actually run them on real, slightly messy, real-world data and see what holds up.

# Dataset 📊 ([Dataset Link](https://github.com/petnica-ml-sci/ml-sci-materials/tree/main/sciml_2026/02_08_intro_task_temp_prediction))

The data comes from the [Open-Meteo Historical Weather API](https://open-meteo.com/), which serves reanalysis weather data (ERA5) for any point on Earth, no API key required. We pulled the last 365 days of daily records for Petnica's coordinates (44.2472° N, 19.9308° E).

You'll find one file, `data/petnica_daily.csv`, one row per day, with these columns:

| Column | What it is |
|---|---|
| `timestamp` | The date |
| `temperature_2m_max` / `_min` / `_mean` | Daily max, min and mean air temperature (°C), measured 2m above ground |
| `apparent_temperature_max` | "Feels like" max temperature, adjusted for wind and humidity |
| `precipitation_sum` | Total precipitation for the day (mm), rain + snow combined |
| `rain_sum` / `snowfall_sum` | The rain and snow components of the total above |
| `precipitation_hours` | How many hours that day had measurable precipitation |
| `wind_speed_10m_max` | Peak sustained wind speed |
| `wind_gusts_10m_max` | Peak wind gust speed |
| `wind_direction_10m_dominant` | Dominant wind direction, in degrees |
| `shortwave_radiation_sum` | Total solar energy reaching the surface that day |
| `et0_fao_evapotranspiration` | Reference evapotranspiration — a standard agricultural measure, driven largely by radiation, temperature and wind |
| `sunrise` / `sunset` | Self-explanatory |
| `daylight_duration` | Length of the day, in seconds |

That's 16 columns and one target hiding among them. Keep in mind that not all of them are pulling their weight, i.e. some are near-duplicates of each other, some barely move the needle, and figuring out which is which is part of the exercise. We're deliberately *not* handing you a pre-cleaned feature set. Look at the correlations, plot things against each other, and decide for yourselves what's worth keeping.

# The Task 🔨

Predict tomorrow's temperature from what you know today (and from however many days back you'd like to look). We'd suggest starting with `temperature_2m_mean` as your target — it's the cleanest single number to chase — but if you want an extra challenge, try `temperature_2m_max` or `temperature_2m_min` too and see if they behave differently.

A couple of things worth keeping in mind, since this is a time series and not a bag of independent rows:

- Don't shuffle and split randomly. Split by time. Train on the earlier months, test on the last stretch, otherwise you're letting your model peek into the future.
- Before reaching for anything clever, build the standard baseline: predict that tomorrow will be the same as today. It's called a persistence baseline, and it's annoyingly hard to beat. Any model you build should be judged against it, not against zero.
- Beyond that, whatever you've got is fair game!

# Have Fun! 👀

Don't worry about getting the perfect score. Care about your methodology! Look at the data, analyze it as much as possible, pick features you deem relevant, and make use of some approach you find plausible.
Plot as many things as you wish, so you can see how the curves move.

Have fun with it. Petnica's weather will keep doing whatever it wants regardless of what your model says, so there's no pressure to be right -- just to be thoughtful about *why* you're wrong.
