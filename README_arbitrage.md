
# Price-Based DC Arbitrage (Non-RL)

Bu paket, **batarya + yük kaydırma (price odaklı) + veri merkezi (IT + soğutma)** bileşenlerini **modüler** şekilde içerir:
- `battery_core.py` – Battery2 etrafında ince sarmalayıcı
- `config_loader.py` – `dc_config.json` yükleyicisi (HVAC parametreleriyle)
- `datacenter_core.py` – DataCenter_ITModel + basit COP(T) modeli
- `load_shift_price.py` – İki eşikli (quantile) greedy arbitraj
- `run_basic.py` – aggregate yük + fiyat ile arbitraj
- `run_detailed.py` – rack utils + CRAC + ambient + fiyat ile arbitraj

## 1) Basic Mode
Giriş CSV'leri:
- **workload.csv**: `timestamp, dc_load_mw`
- **price.csv**: `timestamp, <price_col>`

Çalıştırma:
```bash
python run_basic.py \
  --workload workload.csv \
  --price price.csv \
  --price-col price_eur_per_mwh \
  --fx-to-eur 1.0 \
  --dt 15 \
  --bat-cap 20 --eff-c 0.95 --eff-d 0.95 --c-lim 0.5 --d-lim 0.5 \
  --buy-q 0.30 --sell-q 0.70 \
  --out results_basic.csv
```

> Eğer fiyat kolonu **GBP/MWh** ise, EUR'ye çevirmek için örn. `--fx-to-eur 1.17` gibi bir katsayı ver.

## 2) Detailed Mode
Giriş CSV'leri:
- **rackutils.csv**: `timestamp, rack_1..rack_N, crac_setpoint_c, t_amb_c`
- **price.csv**: `timestamp, <price_col>`
- **dc_config.json**: DC mimarisi + HVAC parametreleri

Çalıştırma:
```bash
python run_detailed.py \
  --rackutils rackutils.csv \
  --price price.csv \
  --price-col price_eur_per_mwh \
  --fx-to-eur 1.0 \
  --dc-config /mnt/data/dc_config.json \
  --dt 15 \
  --bat-cap 20 --eff-c 0.95 --eff-d 0.95 --c-lim 0.5 --d-lim 0.5 \
  --buy-q 0.30 --sell-q 0.70 \
  --out results_detailed.csv
```

## Notlar
- COP modeli basit: `COP = max(1, COP_BASE - K*(T_amb - T_nominal))`. `dc_config.json` içinde HVAC konfigürasyonuyla uyumlu.
- Fiyat eşiklerini kantil ile belirliyoruz; istersen doğrudan mutlak eşik de verecek şekilde fonksiyon genişletilebilir.
- Bu paket, mevcut **Battery2**, **DataCenter_ITModel** ve **DC_Config**’i yeniden kullanır; RL bağımlılığı yoktur.
