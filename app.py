# -*- coding: utf-8 -*-
"""IA_BETA_01_2X 실행 어드바이저 Ver1 — DESIGN_ver1.md 구현.
데이터검증 + 오늘 목표비율 + 보유5칸 입력 + 매매판정(밴드5% + DBMF add-only).
신호=미국 기초자산(yfinance). 상태는 매 실행 과거400일서 재계산(저장 불필요).
"""
import streamlit as st, yfinance as yf, numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")

# ===== LOCKED config =====
TARGET, WIN, FLOOR, CAP, INC = 0.20, 16, 0.20, 1.0, 0.15
GOLD_FRAC, DBMF_FRAC, H_CAP = 0.60, 0.40, 0.60
BAND, MIN_TRADE = 0.05, 0.005

st.set_page_config(page_title="베타2x 리밸런스", layout="centered", initial_sidebar_state="collapsed")
st.title("📈 IA_BETA_01_2X 리밸런스")

@st.cache_data(ttl=1800)
def fetch():
    df = yf.download(["QLD","GLD","DBMF","SGOV"], period="400d", interval="1d",
                     auto_adjust=True, progress=False)
    c = df["Close"] if isinstance(df.columns, pd.MultiIndex) else df
    return c.dropna(how="all").ffill()

def compute_target(px):
    rq = px["QLD"].pct_change()
    vol = (rq.rolling(WIN).std()*np.sqrt(252))
    te = np.clip((TARGET/vol).values, FLOOR, CAP)
    cur = 0.0
    for x in te:
        if np.isnan(x): continue
        if x < cur: cur = x
        elif x - cur > INC: cur = x
    wq = cur
    hb = min(H_CAP, max(0.0, 1 - wq))
    tgt = {"QLD": wq, "금": GOLD_FRAC*hb, "DBMF": DBMF_FRAC*hb, "현금": max(0.0, 1-wq-hb)}
    return tgt, float(vol.iloc[-1]), wq

def decide(hold, deposit, tgt):
    keys = ["QLD","금","DBMF","현금"]
    NAV = sum(hold.values()) + deposit
    if NAV <= 0: return None
    tamt = {a: tgt[a]*NAV for a in keys}
    # (1) DBMF add-only — 새 돈으로만 매수, 매도X
    dbmf_need = tamt["DBMF"] - hold["DBMF"]
    dbmf_buy = min(dbmf_need, deposit) if dbmf_need > 0 else 0.0
    D_rem = deposit - dbmf_buy
    dbmf_final = hold["DBMF"] + dbmf_buy
    # (2) ISA 풀 (QLD/금/현금)
    isa = ["QLD","금","현금"]
    isa_total = NAV - dbmf_final
    rel = np.array([tgt[a] for a in isa]); rel = rel/rel.sum()
    isa_tgt = {a: rel[i]*isa_total for i,a in enumerate(isa)}
    drift = max(abs(hold[a]/NAV - tgt[a]) for a in isa)
    trades = {a: 0.0 for a in keys}; trades["DBMF"] = dbmf_buy
    if drift > BAND:                                   # 밴드 초과 → 전체 리밸런스
        for a in isa: trades[a] = isa_tgt[a] - hold[a]
    elif D_rem > 0:                                    # 밴드 이내 → 입금만 저비중에(매도X)
        under = {a: max(0.0, isa_tgt[a]-hold[a]) for a in isa}
        tot = sum(under.values())
        if tot > 0:
            for a in isa: trades[a] = under[a]/tot * D_rem
        else: trades["현금"] += D_rem
    # (3) 미세거래 필터
    for a in keys:
        if abs(trades[a]) < MIN_TRADE*NAV: trades[a] = 0.0
    cw = {a: hold[a]/NAV for a in keys}
    return dict(NAV=NAV, tamt=tamt, trades=trades, drift=drift, cw=cw, dbmf_buy=dbmf_buy,
                no_trade=all(abs(v) < 1e-6 for v in trades.values()))

# ===== 수집 =====
try:
    px = fetch(); asof = px.index[-1]
    tgt, vol_now, wq = compute_target(px)
    stale = (pd.Timestamp.now()-asof.tz_localize(None)).days if asof.tzinfo else (pd.Timestamp.now()-asof).days
except Exception as e:
    st.error(f"데이터 수집 실패: {e}"); st.stop()

# ===== 상단 배너 =====
st.caption(f"🕒 기준일 **{asof.date()}** (미국장 종가) · 변동성 {vol_now*100:.0f}% → QLD {wq*100:.0f}%")
if stale > 5: st.warning(f"⚠️ 데이터 {stale}일 전 — 휴장/수집지연 확인")

# ===== 🎯 오늘 목표비율 =====
st.subheader("🎯 오늘 목표비율")
c = st.columns(4)
c[0].metric("QLD (ISA)", f"{tgt['QLD']*100:.0f}%")
c[1].metric("금 (ISA)", f"{tgt['금']*100:.0f}%")
c[2].metric("DBMF (해외)", f"{tgt['DBMF']*100:.0f}%")
c[3].metric("현금 (ISA)", f"{tgt['현금']*100:.0f}%")
st.caption(f"근거: QLD 16일변동성 **{vol_now*100:.0f}%** → 원시노출 {min(TARGET/vol_now,CAP):.2f} → 비대칭후 QLD **{wq*100:.0f}%** (나스닥 {wq*2:.2f}배)")

# ===== 📊 데이터 검증 =====
with st.expander("📊 데이터 검증 (야후/증권사와 대조)", expanded=False):
    last = px.iloc[-1]; chg = px.pct_change().iloc[-1]*100
    st.dataframe(pd.DataFrame({"종가($)": last.round(2), "전일대비": chg.round(2).astype(str)+"%"}),
                 use_container_width=True)
    st.caption(f"기준일 {asof.date()} · QLD 16일 실현변동성 {vol_now*100:.1f}%")

# ===== 💼 내 보유 입력 =====
st.subheader("💼 내 보유 입력")
cur_unit = st.radio("통화", ["원화(₩)","달러($)"], horizontal=True, index=0)
st.caption("⚠️ 4자산을 *같은 통화*로 맞춰 입력 (환율 자동변환은 다음 버전)")
g1,g2 = st.columns(2); g3,g4 = st.columns(2)
v_qld  = g1.number_input("[ISA] QLD(2x나스닥)", min_value=0.0, value=0.0, step=100.0, key="v_qld")
v_gold = g2.number_input("[ISA] 금", min_value=0.0, value=0.0, step=100.0, key="v_gold")
v_cash = g3.number_input("[ISA] 현금(파킹)", min_value=0.0, value=0.0, step=100.0, key="v_cash")
v_dbmf = g4.number_input("[해외] DBMF", min_value=0.0, value=0.0, step=100.0, key="v_dbmf")
deposit = st.number_input("💵 오늘 투자할 금액 (적립, 없으면 0)", min_value=0.0, value=0.0, step=100.0, key="dep")

# ===== ⚖️ 매매 판정 =====
st.subheader("⚖️ 매매 판정")
hold = {"QLD":v_qld, "금":v_gold, "DBMF":v_dbmf, "현금":v_cash}
res = decide(hold, deposit, tgt)
if res is None:
    st.info("보유 금액을 입력하면 매매 지시가 나옵니다.")
else:
    if res["no_trade"]:
        st.success(f"🟢 오늘 거래 없음 (최대 드리프트 {res['drift']*100:.1f}% < 5%)")
    else:
        tr = res["trades"]
        st.markdown("**[해외계좌]**")
        st.write(f"· DBMF: {'+'+format(tr['DBMF'],',.0f')+' 매수' if tr['DBMF']>0 else '변동 없음'}  (현재 {res['cw']['DBMF']*100:.0f}% → 목표 {tgt['DBMF']*100:.0f}%)")
        st.markdown("**[ISA계좌]**")
        for a in ["QLD","금","현금"]:
            t = tr[a]; act = f"+{t:,.0f} 매수" if t>0 else (f"{t:,.0f} 매도" if t<0 else "변동 없음")
            st.write(f"· {a}: {act}  (현재 {res['cw'][a]*100:.0f}% → 목표 {tgt[a]*100:.0f}%)")
        if res["drift"]>BAND: st.caption(f"드리프트 {res['drift']*100:.1f}% > 5% → 리밸런스")
    st.caption(f"총자산(입금후) {res['NAV']:,.0f}")

# ===== ▸ 규칙/설정 =====
with st.expander("▸ 규칙 / 설정", expanded=False):
    st.markdown(f"""
**LOCKED 전략**: QLD vol-target(target {TARGET}/win{WIN}/floor{FLOOR}/cap{CAP}/비대칭inc{INC})
· 바스켓 금{GOLD_FRAC:.0%}/DBMF{DBMF_FRAC:.0%}(h≤{H_CAP:.0%}) · 밴드 {BAND:.0%} · 미세거래<{MIN_TRADE:.1%} 무시
**계좌**: ISA=QLD·금·현금(자유 리밸런스) / 해외=DBMF(add-only, 새 돈으로만 매수·매도X)
**신호**: 미국 기초자산(QLD/GLD/DBMF) 종가, 상태는 매 실행 과거400일 재계산
**검증성적**: CAGR~26%, MDD~-23%(실현)/-29%(MC기대), Calmar~1.16
**⚠️ 위험**: 2x — 평시낙폭 -23~29%, 위기 -40%+, 회복 ~18개월. 장기자금만. 폭락에 적립 지속이 핵심.
""")
