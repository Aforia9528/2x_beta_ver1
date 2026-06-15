# 2x Beta Rebalance (IA_BETA_01_2X) — Ver1

QLD vol-target(0.20) + 금/DBMF(60/40) + 현금, 밴드5% + DBMF add-only 일일 리밸런스 어드바이저.

## 실행
```
pip install -r requirements.txt
streamlit run app.py
```
신호=미국 기초자산(QLD/GLD/DBMF, yfinance). 계좌: ISA(QLD·금·현금) / 해외(DBMF).
