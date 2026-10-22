# Changelog

## 2.0.0 - 2026-10-22

A new light-theme dashboard. The 1.0.0 site (scroll-driven story over one map) stays available at tag `v1.0.0`.

### New
- Dashboard layout: sidebar, search (city, sea or storm name), LIVE clock, situation summary.
- Layers panel: pick the satellite layer, set imagery opacity, toggle coastlines, city and sea names, and storm markers.
- Storm card with wind (knots and km/h), position, start and update times, and a GDACS report link. Click a marker or an alert to select a storm.
- Timeline playback: scrub or play the last 6 hours of Himawari images (10-minute steps) or the last 7 days of daily imagery, at 1x, 2x or 4x.
- Current conditions and a 48-hour surface-pressure chart for 13 Indian and nearby cities (Open-Meteo, CC BY 4.0), with a note when pressure is falling fast.
- Live indicators (strongest wind, storms near India, storms worldwide, image age) and a storm-strength bar chart, all drawn from the live feed.
- Reveal-on-scroll for the lower sections and a staggered entrance for the dashboard cards. Reduced motion is respected.

### Kept from 1.0.0
- Real NASA GIBS and Himawari imagery, GDACS active cyclones, honest empty and failure states.
- No-data gaps filled with a dimmed previous-day true-colour layer; black fringe removed from VIIRS tiles.

### Not included
- No AI assistant, no rainfall or wind forecast layers, no 10-day forecast: there is no data source for them in this project.

## 1.0.0 - 2026-10-22

Scroll-driven story over one sticky live satellite map; GDACS cyclone feed; dataset, training and Kaggle pipeline with tests.
