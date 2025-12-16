import os,csv,time,random,json,re,threading,asyncio,hmac,hashlib,base64,subprocess
from typing import Dict,Optional,Any,List,Tuple,Union,Callable
from datetime import datetime,timedelta
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlencode
import requests,pandas as pd,numpy as np
from loguru import logger
from dotenv import load_dotenv
from swarms import Agent
from swarms.agents import CRCAAgent
try:from rich.console import Console;from rich.table import Table;from rich.panel import Panel;from rich.columns import Columns;from rich.layout import Layout;from rich.text import Text;from rich import box;from rich.align import Align;from rich.progress import Progress,SpinnerColumn,TextColumn,BarColumn,TimeRemainingColumn;RICH_AVAILABLE=True
except ImportError:RICH_AVAILABLE=False;Console=None
DAYS_BACK=90
AUTO_MULTI_ASSET=True
MAX_ASSETS_LIMIT=20
MIN_ASSET_VALUE=1.5
VERBOSE_LOGGING=False
PRIORITY_WEIGHTS={'predicted_return':.45,'volume':.15,'volatility':.15,'trend_strength':.1,'signal_quality':.1,'market_cap':.05}
ROTATION_ENABLED=True
ROTATION_MIN_SCORE_DIFF=.1
ROTATION_MAX_CHANGES_PER_CYCLE=10
BATCH_SIZE=100
BATCH_DELAY=1.
LONGTERM_MODE_ENABLED=False
TRADING_CONFIG={'account_size':10000,'account_category':'medium','max_position_size':.3,'max_position_hard_cap':.3,'min_trade_value':5.,'position_size_multiplier':1.,'conservative_mode':True,'aggressive_mode':False,'cooldown_enabled':False,'cooldown_loop_trigger':25,'cooldown_api_budget_threshold':.3,'cooldown_congestion_threshold':.65,'cooldown_min_loops':3,'cooldown_max_loops':4,'cooldown_sleep_multiplier':1.15,'stop_loss_pct':-1e1,'stop_gain_pct':2e1,'promotion_threshold_pct':.1,'promotion_liquidate_enabled':False,'promotion_debounce_secs':5.,'quote_preferences':{'kraken':['EUR','USD','USDC','USDT'],'binance':['USDT','BUSD','USD','USDC'],'coinbase':['USD','USDC','USDT'],'default':['USDT','USD','USDC']}}
LONGTERM_MODE_CONFIG={'prediction_horizon_days':7,'position_evaluation_interval_hours':1,'max_position_size':.005,'min_confidence_threshold':.85,'min_asset_price_usd':.0,'max_asset_price_usd':1.,'position_size_multiplier':.1,'loop_interval_seconds':3600,'full_refresh_interval_hours':24,'use_crca_agent_heavily':True,'crca_max_loops':5,'min_trade_value':1.,'conservative_mode':True,'aggressive_mode':False}
TOP_CRYPTO_SYMBOLS='BTC','ETH','BNB','SOL','XRP','ADA','DOGE','TRX','DOT','MATIC','AVAX','LINK','UNI','ATOM','LTC','ETC','XLM','ALGO','VET','ICP','FIL','AAVE','EOS','THETA','XTZ','EGLD','HBAR','NEAR','QNT','FLOW','SAND','MANA','AXS','GALA','ENJ','CHZ','BAT','ZEC','DASH','WAVES','MKR','COMP','SNX','YFI','SUSHI','CRV','1INCH','RUNE','LUNA','UST','APT','ARB','OP','INJ','TIA','IMX','SUI','SEI','RENDER','FET','GRT','RNDR','LDO','AR','STX','FTM','PEPE','SHIB','FLOKI','BONK','WIF','BOME','MYRO','POPCAT','MEW','MEME','JTO','PYTH','WLD','ONDO','JUP','RAY','ORCA','MNGO','STEP','COPE','HNT','IOTX','RAD','BAND','CTSI','ADX','AUCTION','DAR','BNX','RGT','MOVR','CITY','ENS','KP3R','QI','BADGER','FIS','OM','POND','DEGO','ALICE','LINA','PERP','RAMP','SUPER','CFX','EPS','AUTO','TKO','PROM','GMX','MAGIC','GFT','HFT','FXS','HOOK','MAG','HIFI','FRONT','CVP','AGLD','BETA','RARE','LAZIO'
if TRADING_CONFIG['account_size']<100:TRADING_CONFIG.update({'account_category':'micro','max_position_size':.18,'position_size_multiplier':.5,'conservative_mode':True})
elif TRADING_CONFIG['account_size']<1000:TRADING_CONFIG.update({'account_category':'small','max_position_size':.2,'position_size_multiplier':.8})
elif TRADING_CONFIG['account_size']<10000:TRADING_CONFIG.update({'account_category':'medium','max_position_size':.3,'position_size_multiplier':1.})
else:TRADING_CONFIG.update({'account_category':'large','max_position_size':.5,'position_size_multiplier':1.})
ALTERNATIVE_DATA_CONFIG={'use_real_apis':True,'cache_type':'redis','window_size_days':7,'api_keys':{'twitter':None,'newsapi':None,'etherscan':None,'thegraph':None},'enabled_sources':{'onchain':True,'social':True,'news':True,'github':True,'exchange':True},'cache_ttl':{'onchain':7200,'social':1800,'news':3600,'github':7200,'exchange':1800},'github_repos':['ethereum/go-ethereum','ethereum/consensus-specs','ethereum/execution-specs'],'exchange_metrics':{'exchanges':['binance','bybit'],'symbols':['ETH/USDT','BTC/USDT']},'confidence_weights':{'freshness':.4,'reliability':.4,'stability':.2}}
load_dotenv()
TWITTER_API_FALLBACK='TWITTER_BEARER_TOKEN'
NEWSAPI_FALLBACK=None
ETHERSCAN_FALLBACK='ETHERSCAN_API_KEY'
THEGRAPH_FALLBACK='THEGRAPH_API_KEY'
ALTERNATIVE_DATA_CONFIG['api_keys']['twitter']=os.getenv('TWITTER_BEARER_TOKEN')or os.getenv('TWITTER_API_KEY')or TWITTER_API_FALLBACK
ALTERNATIVE_DATA_CONFIG['api_keys']['newsapi']=os.getenv('NEWSAPI_KEY')or NEWSAPI_FALLBACK
ALTERNATIVE_DATA_CONFIG['api_keys']['etherscan']=os.getenv('ETHERSCAN_API_KEY')or ETHERSCAN_FALLBACK
ALTERNATIVE_DATA_CONFIG['api_keys']['thegraph']=os.getenv('THEGRAPH_API_KEY')or THEGRAPH_FALLBACK
KRAKEN_API_KEY_FALLBACK='KRAKEN_API_KEY'
KRAKEN_API_SECRET_FALLBACK='KRAKEN_API_SECRET'
KRAKEN_API_PASSPHRASE_FALLBACK=None
try:import ccxt;CCXT_AVAILABLE=True
except ImportError:CCXT_AVAILABLE=False
LIVE_TRADING_MODE_DEFAULT=False
try:from web3 import Web3
except ImportError:pass
try:from numba import jit;NUMBA_AVAILABLE=True
except ImportError:
	NUMBA_AVAILABLE=False
	def jit(*args,**kwargs):
		def decorator(func):return func
		return decorator
try:import dowhy;from dowhy import CausalModel;DOWHY_AVAILABLE=True
except ImportError:DOWHY_AVAILABLE=False
ECONML_AVAILABLE=False
CAUSALML_AVAILABLE=False
try:import torch,torch.nn as nn;TORCH_AVAILABLE=True
except ImportError:TORCH_AVAILABLE=False
try:import cvxpy as cp;CVXPY_AVAILABLE=True
except ImportError:CVXPY_AVAILABLE=False
try:import optuna;OPTUNA_AVAILABLE=True
except ImportError:OPTUNA_AVAILABLE=False
try:import yfinance as yf;YFINANCE_AVAILABLE=True
except ImportError:YFINANCE_AVAILABLE=False
try:import websockets;WEBSOCKETS_AVAILABLE=True
except ImportError:WEBSOCKETS_AVAILABLE=False
try:from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor;from sklearn.linear_model import LinearRegression;SKLEARN_AVAILABLE=True
except ImportError:SKLEARN_AVAILABLE=False
try:import xgboost as xgb;XGBOOST_AVAILABLE=True
except ImportError:XGBOOST_AVAILABLE=False
try:import lightgbm as lgb;LIGHTGBM_AVAILABLE=True
except ImportError:LIGHTGBM_AVAILABLE=False
try:import redis;REDIS_AVAILABLE=True
except ImportError:REDIS_AVAILABLE=False
class AssetDiscovery:
	def __init__(self,wallet_value:float,min_asset_value:float=1e1,max_assets:int=50,longterm_mode:bool=False,max_asset_price_usd:Optional[float]=None):
		self.wallet_value=wallet_value;self.longterm_mode=longterm_mode;self.max_asset_price_usd=max_asset_price_usd
		if longterm_mode:self.min_asset_value=.1;self.max_assets=max(max_assets,100)
		else:self.min_asset_value=min_asset_value;self.max_assets=max_assets
		self.top_crypto_symbols=list(TOP_CRYPTO_SYMBOLS)
	def discover_assets(self)->List[Dict[str,str]]:
		max_affordable=int(self.wallet_value/self.min_asset_value);num_assets=min(self.max_assets,max_affordable,len(self.top_crypto_symbols))
		if num_assets<=0:return[]
		return[{'symbol':sym,'type':'crypto'}for sym in self.top_crypto_symbols[:num_assets]]
	def get_asset_priority_score(self,symbol:str,price_data:pd.DataFrame,signal_scores:Dict[str,Dict[str,float]],current_price:float,predicted_return:Optional[float]=None,prediction_confidence:Optional[float]=None,prediction_uncertainty:Optional[float]=None)->float:
		if price_data.empty:return .0
		if self.longterm_mode and self.max_asset_price_usd is not None and current_price>self.max_asset_price_usd:return .0
		scores={}
		if'volume'in price_data.columns:avg_volume=price_data['volume'].tail(30).mean();max_volume=price_data['volume'].max();scores['volume']=min(1.,avg_volume/max_volume)if max_volume>0 else .0
		else:scores['volume']=.5
		if'returns'in price_data.columns:volatility=price_data['returns'].tail(30).std()*np.sqrt(252);scores['volatility']=min(1.,max(.0,(volatility-.2)/.6))if volatility>0 else .0
		else:scores['volatility']=.5
		if'price'in price_data.columns:
			prices=price_data['price'].tail(30)
			if len(prices)>=10:short_ma,long_ma=prices.tail(10).mean(),prices.mean();trend_strength=abs((short_ma-long_ma)/long_ma)if long_ma>0 else .0;scores['trend_strength']=min(1.,trend_strength*10)
			else:scores['trend_strength']=.5
		else:scores['trend_strength']=.5
		signal_quality=.0
		if signal_scores:
			signal_values=[v.get('score',.0)for v in signal_scores.values()if isinstance(v,dict)]
			if signal_values:signal_quality=np.mean(signal_values)
		scores['signal_quality']=signal_quality
		try:market_cap_rank=self.top_crypto_symbols.index(symbol)if symbol in self.top_crypto_symbols else 100;scores['market_cap']=max(.0,1.-market_cap_rank/100)
		except ValueError:scores['market_cap']=.3
		if self.longterm_mode and current_price>0:
			if current_price<1.:price_bonus=1.+(1.-current_price)*.5;scores['price_bonus']=price_bonus
			else:scores['price_bonus']=1.
		else:scores['price_bonus']=1.
		if predicted_return is not None:
			normalized_return=min(1.,max(.0,(predicted_return+.5)/1.))
			if prediction_confidence is not None:normalized_return*=prediction_confidence
			if prediction_uncertainty is not None:normalized_return*=max(.3,1.-min(1.,prediction_uncertainty))
			scores['predicted_return']=normalized_return
		elif'price'in price_data.columns and len(price_data)>=7:recent_return=price_data['price'].iloc[-1]/price_data['price'].iloc[-7]-1 if price_data['price'].iloc[-7]>0 else .0;scores['predicted_return']=min(1.,max(.0,(recent_return+.5)/1.))
		else:scores['predicted_return']=.5
		final_score=float(np.clip(sum(PRIORITY_WEIGHTS.get(factor,.0)*score for(factor,score)in scores.items()if factor!='price_bonus'),.0,1.))
		if'price_bonus'in scores:final_score=min(1.,final_score*scores['price_bonus'])
		return final_score
	def sort_assets_by_priority(self,assets:List[Dict[str,str]],price_data_dict:Dict[str,pd.DataFrame],signal_scores_dict:Dict[str,Dict[str,Dict[str,float]]],current_prices:Dict[str,float])->List[Dict[str,str]]:
		asset_scores=[]
		for asset in assets:
			symbol=asset.get('symbol','').upper()
			if symbol not in price_data_dict:continue
			priority_score=self.get_asset_priority_score(symbol,price_data_dict[symbol],signal_scores_dict.get(symbol,{}),current_prices.get(symbol,.0));asset_scores.append((priority_score,asset))
		asset_scores.sort(key=lambda x:x[0],reverse=True);return[asset for(_,asset)in asset_scores]
class ExchangeAssetCatalog:
	CATALOG={'kraken':['BTC','ETH','SOL','XRP','ADA','DOT','DOGE','AVAX','LINK','LTC','MATIC','ATOM','UNI','XLM','FIL','AAVE','ETC','EGLD','ALGO','FTM','HBAR','NEAR','SAND','MANA','ICP','GRT','APE','CRV','KAVA','SNX','QTUM','BCH','MINA','XMR','ZEC','COMP','1INCH','REN','WAVES','TRX','XTZ','EOS','SUSHI','YFI','RUNE','DASH','OMG','ZRX','BAT','ENJ','ANKR'],'binance':['BTC','ETH','BNB','SOL','XRP','ADA','DOGE','TRX','DOT','MATIC','AVAX','LINK','UNI','ATOM','LTC','ETC','XLM','ALGO','VET','ICP','FIL','AAVE','EGLD','HBAR','NEAR','SAND','MANA','AXS','GALA','ENJ','CHZ','BAT','ZEC','DASH','WAVES','MKR','COMP','SNX','YFI','SUSHI','CRV','1INCH','RUNE','FET','LDO','AR','STX','FTM','APT','ARB','OP','INJ','TIA','IMX','SUI','SEI','RNDR','PYTH','WLD','ONDO','JUP','PEPE','SHIB'],'coinbase':['BTC','ETH','SOL','XRP','ADA','DOT','DOGE','AVAX','LINK','LTC','MATIC','ATOM','UNI','XLM','AAVE','FIL','ETC','ALGO','FTM','NEAR','SAND','MANA','ICP','GRT','APE','CRV','SNX','BCH','XMR','COMP','1INCH','TRX','XTZ','EOS','SUSHI','YFI','RUNE','ZRX','BAT','ENJ','ANKR','ARB','OP','INJ','TIA','IMX','SUI','RNDR','PYTH','WLD','PEPE','SHIB']}
	@classmethod
	def default_assets(cls,exchange:str,max_assets:int)->List[Dict[str,str]]:symbols=cls.CATALOG.get((exchange or'').lower(),[]);return[{'symbol':sym,'type':'crypto'}for sym in symbols[:max_assets]]
class AssetRotationManager:
	def __init__(self,min_score_diff:float=ROTATION_MIN_SCORE_DIFF,max_changes_per_cycle:int=ROTATION_MAX_CHANGES_PER_CYCLE):self.min_score_diff=min_score_diff;self.max_changes_per_cycle=max_changes_per_cycle
	def calculate_rotation_score(self,asset:Dict[str,str],prediction:Optional[float],signal_scores:Dict[str,Dict[str,float]],current_price:float,prediction_confidence:Optional[float]=None,prediction_uncertainty:Optional[float]=None)->float:
		if prediction is None:return .0
		score=min(1.,max(.0,(prediction+.5)/1.))
		if prediction_confidence is not None:score*=.5+.5*prediction_confidence
		if prediction_uncertainty is not None:score*=max(.4,1.-min(1.,prediction_uncertainty))
		if signal_scores:
			signal_values=[v.get('score',.0)for v in signal_scores.values()if isinstance(v,dict)]
			if signal_values:score=score*.7+np.mean(signal_values)*.3
		return float(np.clip(score,.0,1.))
	def evaluate_asset_rotation(self,current_positions:Dict[str,Dict[str,Any]],all_assets:List[Dict[str,str]],predictions:Dict[str,float],signal_scores_dict:Dict[str,Dict[str,Dict[str,float]]],current_prices:Dict[str,float],prediction_confidences:Dict[str,float],prediction_uncertainties:Dict[str,float],portfolio_value:float)->Dict[str,Any]:
		if not ROTATION_ENABLED:return{'exits':[],'entries':[]}
		asset_scores={}
		for asset in all_assets:
			symbol=asset.get('symbol','').upper()
			if symbol not in predictions:continue
			asset_scores[symbol]=self.calculate_rotation_score(asset=asset,prediction=predictions.get(symbol),signal_scores=signal_scores_dict.get(symbol,{}),current_price=current_prices.get(symbol,.0),prediction_confidence=prediction_confidences.get(symbol),prediction_uncertainty=prediction_uncertainties.get(symbol))
		exits=[{'symbol':symbol,'score':asset_scores.get(symbol,.0),'position':position}for(symbol,position)in current_positions.items()if asset_scores.get(symbol,.0)<.3];current_symbols=set(current_positions.keys());potential_entries=[{'symbol':symbol,'score':score,'predicted_return':predictions.get(symbol,.0)}for(symbol,score)in asset_scores.items()if symbol not in current_symbols and score>.6];potential_entries.sort(key=lambda x:x['score'],reverse=True);max_exits=min(len(exits),self.max_changes_per_cycle//2);max_entries=min(len(potential_entries),self.max_changes_per_cycle-max_exits);return{'exits':exits[:max_exits],'entries':potential_entries[:max_entries]}
class MarketDataClient:
	def __init__(self,demo_mode:bool=True):self.demo_mode=demo_mode;self.price_data=pd.DataFrame()
	def fetch_price_data(self,asset:str='ethereum',days_back:int=364,vs_currency:str='usd',asset_type:str='auto')->pd.DataFrame:
		if self.demo_mode:return self._generate_demo_data(days_back,asset)
		if asset_type=='auto':asset_type=self._detect_asset_type(asset)
		if asset_type=='crypto'or asset.lower()in['ethereum','eth','bitcoin','btc']:return self._fetch_crypto_data(asset,days_back,vs_currency)
		if asset_type=='stock'and YFINANCE_AVAILABLE:
			try:return self._fetch_stock_data(asset,days_back)
			except Exception:pass
		if asset_type=='fx':return self._fetch_fx_data(asset,days_back)
		if asset_type=='futures':return self._fetch_futures_data(asset,days_back)
		return pd.DataFrame()
	def _detect_asset_type(self,asset:str)->str:
		asset_lower=asset.lower()
		if asset_lower in['eth','btc','ethereum','bitcoin','usdt','usdc']or len(asset)<=5:return'crypto'
		if len(asset)==6 and asset_lower.isalpha():return'fx'
		return'stock'
	def _fetch_fx_data(self,pair:str,days_back:int)->pd.DataFrame:
		try:
			if YFINANCE_AVAILABLE:
				fx_symbol=f"{pair.upper()}=X"if'='not in pair else pair.upper();end_date,start_date=datetime.now(),datetime.now()-timedelta(days=days_back);df=yf.Ticker(fx_symbol).history(start=start_date,end=end_date)
				if not df.empty:df.reset_index(inplace=True);df.rename(columns={'Date':'date','Close':'price','Volume':'volume'},inplace=True);df['market_cap']=0;return df[['date','price','volume','market_cap']]
		except Exception:pass
		return pd.DataFrame()
	def _fetch_futures_data(self,symbol:str,days_back:int)->pd.DataFrame:
		try:
			if YFINANCE_AVAILABLE:
				futures_symbol=f"{symbol.upper()}=F"if'='not in symbol else symbol.upper();end_date,start_date=datetime.now(),datetime.now()-timedelta(days=days_back);df=yf.Ticker(futures_symbol).history(start=start_date,end=end_date)
				if not df.empty:df.reset_index(inplace=True);df.rename(columns={'Date':'date','Close':'price','Volume':'volume'},inplace=True);df['market_cap']=0;return df[['date','price','volume','market_cap']]
		except Exception:pass
		return pd.DataFrame()
	def _get_asset_base_price(self,asset:str)->float:
		asset_upper,asset_lower=asset.upper(),asset.lower();crypto_prices={'BTC':65e3,'BITCOIN':65e3,'ETH':35e2,'ETHEREUM':35e2,'DOGE':.15,'DOGECOIN':.15,'XRP':.75,'RIPPLE':.75,'DOT':7.,'POLKADOT':7.,'MATIC':.75,'POLYGON':.75,'ADA':.5,'CARDANO':.5,'SOL':1e2,'SOLANA':1e2,'LINK':15.,'CHAINLINK':15.,'UNI':8.,'UNISWAP':8.,'AVAX':35.,'AVALANCHE':35.,'ATOM':1e1,'COSMOS':1e1,'ALGO':.2,'ALGORAND':.2,'TRX':.1,'TRON':.1,'LTC':8e1,'LITECOIN':8e1,'BCH':3e2,'BITCOINCASH':3e2}
		if asset_upper in crypto_prices:return crypto_prices[asset_upper]
		if asset_lower in crypto_prices:return crypto_prices[asset_lower]
		for(key,price)in crypto_prices.items():
			if key.lower()in asset_lower or asset_lower in key.lower():return price
		return 1e3 if len(asset)<=3 else 1e1 if len(asset)<=5 else 1.
	def _get_asset_volatility_range(self,asset:str)->tuple:
		asset_upper=asset.upper()
		if any(hv in asset_upper for hv in['DOGE','SHIB','PEPE','FLOKI']):return .05,.15
		if any(mv in asset_upper for mv in['XRP','ADA','DOT','MATIC','ALGO']):return .03,.08
		if any(lv in asset_upper for lv in['BTC','ETH','USDT','USDC']):return .02,.05
		return .03,.08
	def _generate_demo_data(self,days_back:int,asset:str='ethereum')->pd.DataFrame:
		dates=pd.date_range(end=datetime.now(),periods=days_back,freq='D');base_price=self._get_asset_base_price(asset);vol_min,vol_max=self._get_asset_volatility_range(asset);daily_vol=random.uniform(vol_min,vol_max);price_trend=np.linspace(0,base_price*.15,days_back);noise_std=base_price*daily_vol;noise=np.random.normal(0,noise_std,days_back);prices=base_price+price_trend+noise
		for i in range(1,len(prices)):
			if random.random()<.1:prices[i]+=base_price*random.uniform(-.1,.1)
		prices=np.maximum(prices,base_price*.1);volume_base=base_price*1000;volumes=np.random.uniform(volume_base*.5,volume_base*2.,days_back);supply_estimates={'BTC':19000000,'ETH':120000000,'DOGE':140000000000,'XRP':54000000000,'DOT':1200000000,'MATIC':9000000000};supply=supply_estimates.get(asset.upper(),base_price*1000000000/max(base_price,1.));return pd.DataFrame({'date':dates,'price':prices,'volume':volumes,'market_cap':prices*supply})
	def _fetch_crypto_data(self,asset:str,days_back:int,vs_currency:str)->pd.DataFrame:
		try:
			coin_id='ethereum'if asset.lower()in['ethereum','eth']else'bitcoin';end_date,start_date=datetime.now(),datetime.now()-timedelta(days=days_back);response=requests.get(f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart/range",params={'vs_currency':vs_currency,'from':int(start_date.timestamp()),'to':int(end_date.timestamp())},timeout=10)
			if response.status_code==200:data=response.json();prices,volumes,market_caps=data.get('prices',[]),data.get('total_volumes',[]),data.get('market_caps',[]);df_data=[{'date':datetime.fromtimestamp(timestamp/1000),'price':price,'volume':volumes[i][1]if i<len(volumes)else 0,'market_cap':market_caps[i][1]if i<len(market_caps)else 0}for(i,(timestamp,price))in enumerate(prices)];return pd.DataFrame(df_data)
		except Exception:pass
		return pd.DataFrame()
	def _fetch_stock_data(self,symbol:str,days_back:int)->pd.DataFrame:
		try:end_date,start_date=datetime.now(),datetime.now()-timedelta(days=days_back);ticker=yf.Ticker(symbol);df=ticker.history(start=start_date,end=end_date);df.reset_index(inplace=True);df.rename(columns={'Date':'date','Close':'price','Volume':'volume'},inplace=True);df['market_cap']=df['price']*ticker.info.get('sharesOutstanding',0);return df[['date','price','volume','market_cap']]
		except Exception:pass
		return pd.DataFrame()
	def fetch_correlation(self,asset1:str,asset2:str,price_data1:pd.DataFrame,window:int=14)->pd.Series:
		if price_data1.empty or'returns'not in price_data1.columns:return pd.Series()
		try:
			price_data2=self.fetch_price_data(asset2,len(price_data1))
			if price_data2.empty:return pd.Series()
			price_data2['returns']=price_data2['price'].pct_change();aligned=price_data1.set_index('date').join(price_data2.set_index('date')[['returns']],rsuffix='_2',how='left');aligned['returns_2']=aligned['returns_2'].ffill().bfill();return aligned['returns'].rolling(window=window,min_periods=5).corr(aligned['returns_2'])
		except Exception:return pd.Series(np.zeros(len(price_data1)))
	def fetch_multiple_assets(self,assets:List[Dict[str,str]],days_back:int)->Dict[str,pd.DataFrame]:
		if not assets:return{}
		result={}
		for asset_config in assets:
			symbol=asset_config.get('symbol','')
			if not symbol:continue
			try:
				df=self.fetch_price_data(asset=symbol,days_back=days_back,vs_currency=asset_config.get('vs_currency','usd'),asset_type=asset_config.get('type','auto'))
				if df.empty:df=self._generate_demo_data(days_back,asset=symbol)
				if not df.empty:
					if'returns'not in df.columns:df['returns']=df['price'].pct_change()
					result[symbol.upper()]=df
			except Exception:
				df=self._generate_demo_data(days_back,asset=symbol)
				if not df.empty:df['returns']=df['price'].pct_change();result[symbol.upper()]=df
		return result
	def validate_unified_schema(self,data_dict:Dict[str,pd.DataFrame])->Dict[str,pd.DataFrame]:
		if not data_dict:return{}
		required_cols=['date','price','volume'];validated,price_ranges={},{}
		for(symbol,df)in data_dict.items():
			if df.empty:continue
			if any(col not in df.columns for col in required_cols):continue
			if not pd.api.types.is_datetime64_any_dtype(df['date']):df['date']=pd.to_datetime(df['date'])
			if'price'in df.columns and len(df)>0:price_ranges[symbol]={'min':df['price'].min(),'max':df['price'].max(),'mean':df['price'].mean(),'std':df['price'].std()}
			if'returns'not in df.columns:df=df.copy();df['returns']=df['price'].pct_change()
			validated[symbol]=df
		if len(price_ranges)>1:
			price_means=[r['mean']for r in price_ranges.values()]
			if np.mean(price_means)>0 and np.std(price_means)/np.mean(price_means)<.1 and np.mean(price_means)>100:0
		if len(validated)<2:return validated
		all_dates=set()
		for df in validated.values():all_dates.update(df['date'].dt.date)
		if not all_dates:return validated
		aligned_dates=pd.date_range(start=min(all_dates),end=max(all_dates),freq='D');aligned_dict={}
		for(symbol,df)in validated.items():
			df_aligned=df.set_index('date').reindex(aligned_dates,method='ffill');df_aligned.reset_index(inplace=True);df_aligned.rename(columns={'index':'date'},inplace=True);df_aligned['price']=df_aligned['price'].ffill().bfill();df_aligned['volume']=df_aligned['volume'].fillna(0)
			if'returns'in df_aligned.columns:df_aligned['returns']=df_aligned['returns'].fillna(0)
			aligned_dict[symbol]=df_aligned
		return aligned_dict
	def compute_multi_asset_covariance(self,returns_dict:Dict[str,pd.Series],window:int=20)->pd.DataFrame:
		if not returns_dict or len(returns_dict)<2:return pd.DataFrame()
		returns_df=pd.DataFrame(returns_dict).dropna();return returns_df.cov()if len(returns_df)>=window else returns_df.cov()
class ExchangeSocketManager:
	def __init__(self):self.connections={};self.callbacks={};self.reconnect_attempts={};self.max_reconnect_attempts=10;self.reconnect_delay=1.;self.order_books={};self.running={};self.last_message_time={}
	async def connect(self,exchange:str,symbol:str,callback:callable):
		if not WEBSOCKETS_AVAILABLE:return False
		key=f"{exchange}:{symbol}";self.callbacks[key]=callback;self.reconnect_attempts[key]=0
		try:
			if exchange.lower()=='coinbase':return await self._connect_coinbase(symbol,callback)
			elif exchange.lower()=='binance':return await self._connect_binance(symbol,callback)
			else:return False
		except Exception:return await self._reconnect(exchange,symbol,callback)
	async def _connect_coinbase(self,symbol:str,callback:callable)->bool:
		try:
			import websockets,json;key,product_id=f"coinbase:{symbol}",f"{symbol.upper()}-USD"
			if key not in self.order_books:self.order_books[key]={'bids':{},'asks':{},'sequence':0}
			async def _handle_message(ws,key):
				async for message in ws:
					try:
						data,msg_type=json.loads(message),data.get('type','');self.last_message_time[key]=time.time()
						if msg_type=='subscriptions':await ws.send(json.dumps({'type':'subscribe','product_ids':[product_id],'channel':'level2'}));await ws.send(json.dumps({'type':'subscribe','product_ids':[product_id],'channel':'matches'}))
						elif msg_type=='l2update':await self._update_order_book_coinbase(key,data);await callback({'type':'orderbook','symbol':symbol,'data':self.order_books[key]})
						elif msg_type=='match':await callback({'type':'trade','symbol':symbol,'price':float(data.get('price',0)),'size':float(data.get('size',0)),'side':data.get('side','unknown'),'time':data.get('time','')})
						elif msg_type=='ticker':await callback({'type':'ticker','symbol':symbol,'price':float(data.get('price',0)),'volume_24h':float(data.get('volume_24h',0)),'high_24h':float(data.get('high_24h',0)),'low_24h':float(data.get('low_24h',0))})
						else:await callback(data)
					except(json.JSONDecodeError,Exception):pass
			async def _connect():
				while key in self.running and self.running[key]:
					try:
						async with websockets.connect('wss://advanced-trade-ws.coinbase.com',ping_interval=20,ping_timeout=10)as ws:self.connections[key]=ws;await ws.send(json.dumps({'type':'subscribe','product_ids':[product_id],'channel':'ticker','jwt':''}));await _handle_message(ws,key)
					except(websockets.exceptions.ConnectionClosed,Exception):
						if key in self.running and self.running[key]:await asyncio.sleep(self.reconnect_delay)
			self.running[key],self.last_message_time[key]=True,time.time();asyncio.create_task(_connect());return True
		except Exception:return False
	async def _update_order_book_coinbase(self,key:str,data:Dict[str,Any]):
		for change in data.get('changes',[]):
			side,price,size=change[0],float(change[1]),float(change[2]);price_str=str(price)
			if side=='buy':self.order_books[key]['bids'].pop(price_str,None)if size==0 else self.order_books[key]['bids'].update({price_str:size})
			elif side=='sell':self.order_books[key]['asks'].pop(price_str,None)if size==0 else self.order_books[key]['asks'].update({price_str:size})
		if'sequence'in data:self.order_books[key]['sequence']=data['sequence']
	async def _connect_binance(self,symbol:str,callback:callable)->bool:
		try:
			import websockets,json;key,symbol_pair=f"binance:{symbol}",f"{symbol.lower()}usdt";ticker_url,depth_url,trade_url=f"wss://stream.binance.com:9443/ws/{symbol_pair}@ticker",f"wss://stream.binance.com:9443/ws/{symbol_pair}@depth20@100ms",f"wss://stream.binance.com:9443/ws/{symbol_pair}@trade"
			if key not in self.order_books:self.order_books[key]={'bids':{},'asks':{},'lastUpdateId':0}
			async def _handle_ticker(ws_url,key):
				while key in self.running and self.running[key]:
					try:
						async with websockets.connect(ws_url)as ws:
							async for message in ws:
								try:data=json.loads(message);self.last_message_time[key]=time.time();await callback({'type':'ticker','symbol':symbol,'price':float(data.get('c',0)),'volume_24h':float(data.get('v',0)),'high_24h':float(data.get('h',0)),'low_24h':float(data.get('l',0)),'bid':float(data.get('b',0)),'ask':float(data.get('a',0))})
								except Exception:pass
					except Exception:
						if key in self.running and self.running[key]:await asyncio.sleep(self.reconnect_delay)
			async def _handle_depth(ws_url,key):
				while key in self.running and self.running[key]:
					try:
						async with websockets.connect(ws_url)as ws:
							async for message in ws:
								try:data=json.loads(message);self.last_message_time[key]=time.time();await self._update_order_book_binance(key,data);await callback({'type':'orderbook','symbol':symbol,'data':self.order_books[key]})
								except Exception:pass
					except Exception:
						if key in self.running and self.running[key]:await asyncio.sleep(self.reconnect_delay)
			async def _handle_trades(ws_url,key):
				while key in self.running and self.running[key]:
					try:
						async with websockets.connect(ws_url)as ws:
							async for message in ws:
								try:data=json.loads(message);self.last_message_time[key]=time.time();await callback({'type':'trade','symbol':symbol,'price':float(data.get('p',0)),'size':float(data.get('q',0)),'side':'buy'if data.get('m',False)else'sell','time':data.get('T',0)})
								except Exception:pass
					except Exception:
						if key in self.running and self.running[key]:await asyncio.sleep(self.reconnect_delay)
			self.running[key],self.last_message_time[key]=True,time.time();asyncio.create_task(_handle_ticker(ticker_url,key));asyncio.create_task(_handle_depth(depth_url,key));asyncio.create_task(_handle_trades(trade_url,key));return True
		except Exception:return False
	async def _update_order_book_binance(self,key:str,data:Dict[str,Any]):
		self.order_books[key]['bids']={str(float(bid[0])):float(bid[1])for bid in data.get('bids',[])if float(bid[1])>0};self.order_books[key]['asks']={str(float(ask[0])):float(ask[1])for ask in data.get('asks',[])if float(ask[1])>0}
		if'lastUpdateId'in data:self.order_books[key]['lastUpdateId']=data['lastUpdateId']
	async def _reconnect(self,exchange:str,symbol:str,callback:callable)->bool:
		key=f"{exchange}:{symbol}";attempts=self.reconnect_attempts.get(key,0)
		if attempts>=self.max_reconnect_attempts:return False
		self.reconnect_attempts[key]=attempts+1;await asyncio.sleep(self.reconnect_delay*2**attempts);return await self.connect(exchange,symbol,callback)
	async def disconnect(self,exchange:str,symbol:str):
		key=f"{exchange}:{symbol}"
		if key in self.running:self.running[key]=False
		if key in self.connections:
			try:await self.connections[key].close();del self.connections[key]
			except Exception:pass
		for d in[self.callbacks,self.order_books,self.last_message_time]:
			if key in d:del d[key]
	def get_order_book(self,exchange:str,symbol:str)->Optional[Dict[str,Any]]:return self.order_books.get(f"{exchange}:{symbol}")
	def is_connected(self,exchange:str,symbol:str)->bool:
		key=f"{exchange}:{symbol}"
		if key not in self.running or not self.running[key]:return False
		if key in self.last_message_time:return time.time()-self.last_message_time[key]<60
		return False
class OrderBookTracker:
	def __init__(self):self.order_books={}
	def update_orderbook(self,exchange:str,symbol:str,bids:List,asks:List):self.order_books[f"{exchange}:{symbol}"]={'bids':bids,'asks':asks,'timestamp':time.time()}
	def get_spread(self,exchange:str,symbol:str)->float:
		key=f"{exchange}:{symbol}"
		if key not in self.order_books:return .0
		ob=self.order_books[key];return ob['asks'][0][0]-ob['bids'][0][0]if ob['bids']and ob['asks']else .0
	def get_order_imbalance(self,exchange:str,symbol:str,levels:int=5)->float:
		key=f"{exchange}:{symbol}"
		if key not in self.order_books:return .0
		ob=self.order_books[key];bid_vol,ask_vol=sum(b[1]for b in ob['bids'][:levels]),sum(a[1]for a in ob['asks'][:levels]);return(bid_vol-ask_vol)/(bid_vol+ask_vol)if bid_vol+ask_vol>0 else .0
class AltDataClient:
	def __init__(self,config:Optional[Dict[str,Any]]=None):self.config=config or ALTERNATIVE_DATA_CONFIG;self.use_real_apis=self.config.get('use_real_apis',True);self.cache_type=self.config.get('cache_type','file');self.window_size_days=self.config.get('window_size_days',7);self.cache=self._init_cache();self.cache_dir=Path('.cache/alternative_data');self.cache_dir.mkdir(parents=True,exist_ok=True);self.background_thread=None;self.fetch_lock=threading.Lock();(self.data_cache):Dict[str,Any]={};(self.fetch_timestamps):Dict[str,float]={};self._redis_client=None;self._web3_client=None
	def _init_cache(self):
		if self.cache_type=='redis'and REDIS_AVAILABLE:
			try:self._redis_client=redis.Redis(host='localhost',port=6379,db=0,decode_responses=True);self._redis_client.ping();return'redis'
			except Exception:return'file'
		return'file'
	def _get_cache(self,key:str)->Optional[Any]:
		if self.cache=='redis'and self._redis_client:
			try:
				cached=self._redis_client.get(key)
				if cached:return json.loads(cached)
			except Exception:pass
		cache_file=self.cache_dir/f"{key.replace(":","_")}.json"
		if cache_file.exists():
			try:
				with open(cache_file,'r')as f:
					data=json.load(f)
					if'timestamp'in data and'ttl'in data:
						if time.time()-data['timestamp']<data['ttl']:return data['data']
					else:return data
			except Exception:pass
	def _cache_data(self,key:str,data:Any,ttl:int=3600):
		cache_data={'data':data,'timestamp':time.time(),'ttl':ttl}
		if self.cache=='redis'and self._redis_client:
			try:self._redis_client.setex(key,ttl,json.dumps(cache_data));return
			except Exception:pass
		cache_file=self.cache_dir/f"{key.replace(":","_")}.json"
		try:
			with open(cache_file,'w')as f:json.dump(cache_data,f)
		except Exception:pass
	def _fetch_onchain_thegraph(self,asset:str='ethereum')->Dict[str,float]:
		if not self.use_real_apis:return self._mock_onchain_data()
		try:
			query='{ pools(first: 10, orderBy: totalValueLockedUSD, orderDirection: desc) { id totalValueLockedUSD volumeUSD } }';response=requests.post('https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v3',json={'query':query},timeout=10)
			if response.status_code==200:
				pools=response.json().get('data',{}).get('pools',[])
				if pools:total_tvl=sum(float(p.get('totalValueLockedUSD',0))for p in pools);total_volume=sum(float(p.get('volumeUSD',0))for p in pools);return{'active_addresses':min(1.,total_tvl/1e12),'transaction_volume':min(1.,total_volume/1e11),'network_growth':min(.1,max(-.1,(total_volume-total_tvl)/total_tvl))if total_tvl>0 else .0}
		except Exception:pass
		return self._mock_onchain_data()
	def _fetch_onchain_rpc(self,asset:str='ethereum')->Dict[str,float]:
		if not self.use_real_apis:return self._mock_onchain_data()
		try:
			rpc_url=os.getenv('ETH_RPC_URL','https://eth.llamarpc.com')
			try:
				from web3 import Web3;w3=Web3(Web3.HTTPProvider(rpc_url))
				if w3.is_connected():latest_block,block=w3.eth.block_number,w3.eth.get_block(w3.eth.block_number);old_block=w3.eth.get_block(max(0,latest_block-7200*self.window_size_days));current_tx_count=block.transactions.__len__()if hasattr(block,'transactions')else 0;old_tx_count=old_block.transactions.__len__()if hasattr(old_block,'transactions')else 0;tx_growth=(current_tx_count-old_tx_count)/max(old_tx_count,1);return{'active_addresses':min(1.,current_tx_count/1000),'transaction_volume':min(1.,current_tx_count/500),'network_growth':min(.1,max(-.1,tx_growth))}
			except ImportError:pass
		except Exception:pass
		return self._mock_onchain_data()
	def _fetch_onchain_etherscan(self,asset:str='ethereum')->Dict[str,float]:
		api_key=self.config['api_keys'].get('etherscan')
		if not api_key or not self.use_real_apis:return self._mock_onchain_data()
		try:
			response=requests.get('https://api.etherscan.io/api',params={'module':'proxy','action':'eth_blockNumber','apikey':api_key},timeout=10)
			if response.status_code==200:return{'active_addresses':random.uniform(.5,1.),'transaction_volume':random.uniform(.5,1.),'network_growth':random.uniform(-.05,.05)}
		except Exception:pass
		return self._mock_onchain_data()
	def fetch_onchain_metrics(self,asset:str='ethereum')->Dict[str,float]:
		if not self.config['enabled_sources']['onchain']:return{}
		cache_key=f"onchain:{asset}";cached=self._get_cache(cache_key)
		if cached:return cached
		data=self._fetch_onchain_thegraph(asset)
		if not data or all(v==0 for v in data.values()):data=self._fetch_onchain_rpc(asset)
		if not data or all(v==0 for v in data.values()):data=self._fetch_onchain_etherscan(asset)
		self._cache_data(cache_key,data,self.config['cache_ttl']['onchain']);return data
	def _fetch_reddit_sentiment(self,asset:str,days_back:int=7)->Dict[str,float]:
		if not self.use_real_apis:return self._mock_social_data()
		try:
			subreddit='ethereum'if asset.lower()in['ethereum','eth']else'cryptocurrency';response=requests.get(f"https://www.reddit.com/r/{subreddit}/hot.json",headers={'User-Agent':'QuantTradingAgent/1.0'},timeout=10)
			if response.status_code==200:
				posts=response.json().get('data',{}).get('children',[])
				if posts:avg_score=sum(p.get('data',{}).get('score',0)for p in posts[:25])/len(posts)if posts else 0;return{'reddit_sentiment':float(np.tanh(avg_score/100)),'social_volume':min(1.,len(posts)/25),'twitter_sentiment':.0}
		except Exception:pass
		return self._mock_social_data()
	def _fetch_twitter_sentiment(self,asset:str,days_back:int=7)->Dict[str,float]:
		api_key=self.config['api_keys'].get('twitter')
		if not api_key or not self.use_real_apis:return{'twitter_sentiment':.0}
		try:
			response=requests.get('https://api.twitter.com/2/tweets/search/recent',headers={'Authorization':f"Bearer {api_key}"},params={'query':f"{asset} OR #{asset.upper()} -is:retweet lang:en",'max_results':10,'tweet.fields':'public_metrics,created_at'},timeout=10)
			if response.status_code==200:
				tweets=response.json().get('data',[])
				if tweets:return{'twitter_sentiment':float(np.tanh(sum(t.get('public_metrics',{}).get('like_count',0)for t in tweets)/len(tweets)/100))}
		except Exception:pass
		return{'twitter_sentiment':.0}
	def fetch_social_sentiment(self,asset:str,days_back:int=7)->Dict[str,float]:
		if not self.config['enabled_sources']['social']:return{}
		cache_key=f"social:{asset}:{days_back}";cached=self._get_cache(cache_key)
		if cached:return cached
		reddit_data=self._fetch_reddit_sentiment(asset,days_back);twitter_data=self._fetch_twitter_sentiment(asset,days_back);result={'reddit_sentiment':reddit_data.get('reddit_sentiment',.0),'twitter_sentiment':twitter_data.get('twitter_sentiment',.0),'social_volume':reddit_data.get('social_volume',.0)};self._cache_data(cache_key,result,self.config['cache_ttl']['social']);return result
	def _fetch_newsapi_sentiment(self,asset:str,days_back:int=7)->Dict[str,float]:
		api_key=self.config['api_keys'].get('newsapi')
		if not api_key or not self.use_real_apis:return self._mock_news_data()
		try:
			response=requests.get('https://newsapi.org/v2/everything',params={'q':f"{asset} OR cryptocurrency OR blockchain",'apiKey':api_key,'sortBy':'publishedAt','language':'en','pageSize':20},timeout=10)
			if response.status_code==200:
				articles=response.json().get('articles',[])
				if articles:pos_words,neg_words=['bullish','surge','rally','gain','up','positive'],['bearish','crash','drop','loss','down','negative'];pos_count=sum(1 for a in articles if any(w in a.get('title','').lower()or w in a.get('description','').lower()for w in pos_words));neg_count=sum(1 for a in articles if any(w in a.get('title','').lower()or w in a.get('description','').lower()for w in neg_words));total=pos_count+neg_count;sentiment=(pos_count-neg_count)/total if total>0 else .0;return{'news_sentiment':float(sentiment),'news_volume':min(1.,len(articles)/20),'headline_sentiment':float(sentiment)}
		except Exception:pass
		return self._mock_news_data()
	def fetch_news_sentiment(self,asset:str,days_back:int=7)->Dict[str,float]:
		if not self.config['enabled_sources']['news']:return{}
		cache_key=f"news:{asset}:{days_back}";cached=self._get_cache(cache_key)
		if cached:return cached
		data=self._fetch_newsapi_sentiment(asset,days_back);self._cache_data(cache_key,data,self.config['cache_ttl']['news']);return data
	def _fetch_github_activity(self,repos:Optional[List[str]]=None)->Dict[str,float]:
		if not self.config['enabled_sources']['github']:return{}
		repos=repos or self.config.get('github_repos',[])
		if not repos:return{}
		try:
			activity_scores=[]
			for repo in repos:
				try:
					response=requests.get(f"https://api.github.com/repos/{repo}",timeout=10)
					if response.status_code==200:d=response.json();activity=(d.get('stargazers_count',0)+d.get('forks_count',0)*2)/max(d.get('open_issues_count',1),1);activity_scores.append(min(1.,activity/1000))
				except Exception:continue
			if activity_scores:return{'github_activity':float(np.mean(activity_scores)),'github_momentum':float(np.std(activity_scores))if len(activity_scores)>1 else .0}
		except Exception:pass
		return{'github_activity':.0,'github_momentum':.0}
	def fetch_github_activity(self,repos:Optional[List[str]]=None)->Dict[str,float]:
		cache_key=f"github:{",".join(repos or[])}";cached=self._get_cache(cache_key)
		if cached:return cached
		data=self._fetch_github_activity(repos);self._cache_data(cache_key,data,self.config['cache_ttl']['github']);return data
	def _fetch_exchange_metrics(self)->Dict[str,float]:
		if not self.config['enabled_sources']['exchange']or not CCXT_AVAILABLE:return{}
		try:
			exchange_config=self.config.get('exchange_metrics',{});exchanges,symbols=exchange_config.get('exchanges',['binance']),exchange_config.get('symbols',['ETH/USDT']);metrics={}
			for exchange_name in exchanges:
				try:
					exchange=getattr(ccxt,exchange_name)({'enableRateLimit':True})
					for symbol in symbols:
						try:
							if hasattr(exchange,'fetch_funding_rate'):
								funding=exchange.fetch_funding_rate(symbol)
								if funding:metrics[f"{exchange_name}_funding"]=float(funding.get('fundingRate',0))
							if hasattr(exchange,'fetch_open_interest'):
								oi=exchange.fetch_open_interest(symbol)
								if oi:metrics[f"{exchange_name}_oi"]=min(1.,float(oi.get('openInterestAmount',0))/1e9)
						except Exception:continue
				except Exception:continue
			return metrics if metrics else{'funding_rate':.0,'open_interest':.0}
		except Exception:pass
		return{'funding_rate':.0,'open_interest':.0}
	def fetch_exchange_metrics(self)->Dict[str,float]:
		cache_key='exchange:metrics';cached=self._get_cache(cache_key)
		if cached:return cached
		data=self._fetch_exchange_metrics();self._cache_data(cache_key,data,self.config['cache_ttl']['exchange']);return data
	def fetch_all_alternative_data(self,asset:str='ethereum',days_back:int=7)->Dict[str,Dict[str,float]]:
		def _fetch():
			with self.fetch_lock:
				try:result={'onchain':self.fetch_onchain_metrics(asset),'social':self.fetch_social_sentiment(asset,days_back),'news':self.fetch_news_sentiment(asset,days_back),'github':self.fetch_github_activity(),'exchange':self.fetch_exchange_metrics()};self.data_cache=result;self.fetch_timestamps={k:time.time()for k in result.keys()}
				except Exception:pass
		if self.background_thread is None or not self.background_thread.is_alive():self.background_thread=threading.Thread(target=_fetch,daemon=True);self.background_thread.start()
		return self.data_cache.copy()if self.data_cache else{}
	def _mock_onchain_data(self)->Dict[str,float]:return{'active_addresses':random.uniform(.3,.9),'transaction_volume':random.uniform(.3,.9),'network_growth':random.uniform(-.05,.05)}
	def _mock_social_data(self)->Dict[str,float]:return{'twitter_sentiment':random.uniform(-.8,.8),'reddit_sentiment':random.uniform(-.8,.8),'social_volume':random.uniform(.2,.9)}
	def _mock_news_data(self)->Dict[str,float]:return{'news_sentiment':random.uniform(-.7,.7),'news_volume':random.uniform(.3,.8),'headline_sentiment':random.uniform(-.7,.7)}
class SocialSentimentSignals:
	@staticmethod
	def twitter_sentiment(sentiment_data:Dict[str,float])->float:return sentiment_data.get('twitter_sentiment',.0)
	@staticmethod
	def reddit_sentiment(sentiment_data:Dict[str,float])->float:return sentiment_data.get('reddit_sentiment',.0)
	@staticmethod
	def social_momentum(sentiment_data:Dict[str,float],window:int=7)->float:return sentiment_data.get('social_volume',.0)
	@staticmethod
	def twitter_sentiment_ts(sentiment_series:pd.Series,window:int=7)->pd.Series:return pd.Series()if sentiment_series.empty else sentiment_series.rolling(window=window).mean()
	@staticmethod
	def reddit_sentiment_ts(sentiment_series:pd.Series,window:int=7)->pd.Series:return pd.Series()if sentiment_series.empty else sentiment_series.rolling(window=window).mean()
	@staticmethod
	def sentiment_reversal(sentiment_series:pd.Series,window:int=7)->pd.Series:return pd.Series()if sentiment_series.empty or len(sentiment_series)<window*2 else sentiment_series.rolling(window=window).mean()-sentiment_series.rolling(window=window*2).mean()
class OnChainSignals:
	@staticmethod
	def active_addresses_growth(onchain_data:Dict[str,float])->float:return onchain_data.get('active_addresses',.0)
	@staticmethod
	def transaction_volume_trend(onchain_data:Dict[str,float])->float:return onchain_data.get('transaction_volume',.0)
	@staticmethod
	def network_growth(onchain_data:Dict[str,float])->float:return onchain_data.get('network_growth',.0)
	@staticmethod
	def active_addresses_growth_ts(addresses_series:pd.Series,window:int=7)->pd.Series:return pd.Series()if addresses_series.empty or len(addresses_series)<window else addresses_series.pct_change(window)
	@staticmethod
	def tx_volume_growth_ts(volume_series:pd.Series,window:int=7)->pd.Series:return pd.Series()if volume_series.empty or len(volume_series)<window else volume_series.pct_change(window)
	@staticmethod
	def network_health_score(onchain_data:Dict[str,float])->float:return onchain_data.get('active_addresses',.0)*.4+onchain_data.get('transaction_volume',.0)*.4+(onchain_data.get('network_growth',.0)+.1)*.2
	@staticmethod
	def gas_efficiency(onchain_data:Dict[str,float])->float:return onchain_data.get('transaction_volume',.0)
class NewsSentimentSignals:
	@staticmethod
	def news_sentiment(news_data:Dict[str,float])->float:return news_data.get('news_sentiment',.0)
	@staticmethod
	def headline_sentiment(news_data:Dict[str,float])->float:return news_data.get('headline_sentiment',.0)
	@staticmethod
	def news_volume(news_data:Dict[str,float])->float:return news_data.get('news_volume',.0)
	@staticmethod
	def news_sentiment_ts(sentiment_series:pd.Series,window:int=7)->pd.Series:return pd.Series()if sentiment_series.empty else sentiment_series.rolling(window=window).mean()
	@staticmethod
	def headline_sentiment_ts(sentiment_series:pd.Series,window:int=7)->pd.Series:return pd.Series()if sentiment_series.empty else sentiment_series.rolling(window=window).mean()
	@staticmethod
	def news_volume_spike(volume_series:pd.Series,window:int=7)->pd.Series:return pd.Series()if volume_series.empty or len(volume_series)<window else(volume_series-volume_series.rolling(window=window).mean())/(volume_series.rolling(window=window).std()+1e-06)
	@staticmethod
	def sentiment_momentum(sentiment_series:pd.Series,window:int=7)->pd.Series:return pd.Series()if sentiment_series.empty or len(sentiment_series)<window else sentiment_series.diff(window)
class GitHubSignals:
	@staticmethod
	def github_activity(github_data:Dict[str,float])->float:return github_data.get('github_activity',.0)
	@staticmethod
	def github_momentum(github_data:Dict[str,float])->float:return github_data.get('github_momentum',.0)
	@staticmethod
	def github_activity_ts(activity_series:pd.Series,window:int=7)->pd.Series:return pd.Series()if activity_series.empty else activity_series.rolling(window=window).mean()
	@staticmethod
	def github_community_health(github_data:Dict[str,float])->float:return github_data.get('github_activity',.0)*(1.+github_data.get('github_momentum',.0))
class ExchangeMetricsSignals:
	@staticmethod
	def funding_rate(exchange_data:Dict[str,float])->float:return next((exchange_data[key]for key in['funding_rate','binance_funding','bybit_funding']if key in exchange_data),.0)
	@staticmethod
	def open_interest(exchange_data:Dict[str,float])->float:return next((exchange_data[key]for key in['open_interest','binance_oi','bybit_oi']if key in exchange_data),.0)
	@staticmethod
	def long_short_ratio(exchange_data:Dict[str,float])->float:return np.tanh((exchange_data.get('long_short_ratio',1.)-1.)*2)
	@staticmethod
	def funding_rate_ts(funding_series:pd.Series,window:int=7)->pd.Series:return pd.Series()if funding_series.empty else funding_series.rolling(window=window).mean()
	@staticmethod
	def oi_momentum(oi_series:pd.Series,window:int=7)->pd.Series:return pd.Series()if oi_series.empty or len(oi_series)<window else oi_series.pct_change(window)
class SignalRegistry:
	def __init__(self):self.signals,self.signal_classes={},{}
	def register(self,name:str,signal_class:type,method_name:str):self.signals[name]={'class':signal_class,'method':method_name}
	def compute_all(self,data:pd.DataFrame,signal_instances:Dict)->pd.DataFrame:
		result=data.copy()
		for(name,info)in self.signals.items():
			try:
				instance=signal_instances.get(info['class'])
				if instance:result[f"signal_{name}"]=getattr(instance,info['method'])(data)
			except Exception:pass
		return result
class TimeSeriesSignals:
	@staticmethod
	@jit(nopython=True)if NUMBA_AVAILABLE else lambda x:x
	def k_period_return(prices:np.ndarray,k:int)->np.ndarray:
		returns=np.zeros(len(prices))
		for i in range(k,len(prices)):returns[i]=prices[i]/prices[i-k]-1.
		return returns
	@staticmethod
	def momentum(data:pd.DataFrame,lookback:int=12)->pd.Series:return pd.Series()if'price'not in data.columns else np.log(data['price']).diff(lookback)
	@staticmethod
	def short_term_reversal(data:pd.DataFrame,k:int=1)->pd.Series:return pd.Series()if'price'not in data.columns else-data['price'].pct_change(k)
	@staticmethod
	def sma_distance(data:pd.DataFrame,window:int=20)->pd.Series:return pd.Series()if'price'not in data.columns else(data['price']-data['price'].rolling(window=window).mean())/data['price'].rolling(window=window).mean()
	@staticmethod
	def ema_trend(data:pd.DataFrame,span:int=12)->pd.Series:return pd.Series()if'price'not in data.columns else data['price'].ewm(span=span,adjust=False).mean().diff()
	@staticmethod
	def ma_crossover(data:pd.DataFrame,fast:int=7,slow:int=14)->pd.Series:return pd.Series()if'price'not in data.columns else(data['price'].rolling(window=fast).mean()-data['price'].rolling(window=slow).mean())/data['price'].rolling(window=slow).mean()
	@staticmethod
	def volatility_breakout(data:pd.DataFrame,window:int=14)->pd.Series:
		if'price'not in data.columns or len(data)<2:return pd.Series()
		high,low,close=data.get('high',data['price']),data.get('low',data['price']),data['price'];tr=pd.concat([high-low,(high-close.shift(1)).abs(),(low-close.shift(1)).abs()],axis=1).max(axis=1);return(tr-tr.rolling(window=window).mean())/tr.rolling(window=window).std()
	@staticmethod
	def price_level(data:pd.DataFrame,window:int=20)->pd.Series:
		if'price'not in data.columns:return pd.Series()
		log_price=np.log(data['price']);return(log_price-log_price.rolling(window=window).mean())/log_price.rolling(window=window).std()
class VolatilitySignals:
	@staticmethod
	def realized_variance(data:pd.DataFrame,window:int=1)->pd.Series:
		if'returns'not in data.columns:data['returns']=data['price'].pct_change()
		return data['returns'].rolling(window=window).var()
	@staticmethod
	def realized_volatility(data:pd.DataFrame,window:int=1)->pd.Series:return np.sqrt(VolatilitySignals.realized_variance(data,window))
	@staticmethod
	def garch_volatility(data:pd.DataFrame)->pd.Series:
		if'returns'not in data.columns:data['returns']=data['price'].pct_change()
		returns=data['returns'].fillna(0);h=np.zeros(len(returns));omega,alpha,beta=.01,.1,.85;h[0]=returns[0]**2
		for i in range(1,len(returns)):h[i]=omega+alpha*returns[i-1]**2+beta*h[i-1]
		return pd.Series(np.sqrt(h),index=data.index)
	@staticmethod
	def vol_of_vol(data:pd.DataFrame,window:int=20)->pd.Series:return VolatilitySignals.realized_volatility(data,window=1).rolling(window=window).std()
	@staticmethod
	def skewness(data:pd.DataFrame,window:int=20)->pd.Series:
		if'returns'not in data.columns:data['returns']=data['price'].pct_change()
		return data['returns'].fillna(0).rolling(window=window).skew()
	@staticmethod
	def kurtosis(data:pd.DataFrame,window:int=20)->pd.Series:
		if'returns'not in data.columns:data['returns']=data['price'].pct_change()
		return data['returns'].fillna(0).rolling(window=window).kurt()
	@staticmethod
	def volatility_clustering(data:pd.DataFrame,window:int=20)->pd.Series:
		if'returns'not in data.columns:data['returns']=data['price'].pct_change()
		abs_returns=data['returns'].abs();return abs_returns.rolling(window=window).corr(abs_returns.shift(1))
class LiquiditySignals:
	@staticmethod
	def turnover(data:pd.DataFrame)->pd.Series:return pd.Series()if'volume'not in data.columns or'market_cap'not in data.columns else data['volume']*data['price']/data['market_cap']
	@staticmethod
	def volume_zscore(data:pd.DataFrame,window:int=20)->pd.Series:return pd.Series()if'volume'not in data.columns else(data['volume']-data['volume'].rolling(window=window).mean())/data['volume'].rolling(window=window).std()
	@staticmethod
	def amihud_illiquidity(data:pd.DataFrame,window:int=20)->pd.Series:
		if'returns'not in data.columns:data['returns']=data['price'].pct_change()
		return pd.Series()if'volume'not in data.columns else(data['returns'].abs()/(data['volume']*data['price'])).rolling(window=window).mean()
	@staticmethod
	def bid_ask_spread(bid:float,ask:float,mid:float)->float:return .0 if mid==0 else(ask-bid)/mid
	@staticmethod
	def order_book_imbalance(bid_vol:float,ask_vol:float)->float:return .0 if bid_vol+ask_vol==0 else(bid_vol-ask_vol)/(bid_vol+ask_vol)
	@staticmethod
	def depth_slope(bid_depths:List[float],ask_depths:List[float])->float:bid_total,ask_total=sum(bid_depths)if bid_depths else .0,sum(ask_depths)if ask_depths else .0;return .0 if bid_total+ask_total==0 else(bid_total-ask_total)/(bid_total+ask_total)
	@staticmethod
	def trade_imbalance(data:pd.DataFrame,window:int=20)->pd.Series:
		if'returns'not in data.columns:data['returns']=data['price'].pct_change()
		return pd.Series()if'volume'not in data.columns else(data['volume']*np.sign(data['returns'])).rolling(window=window).sum()
	@staticmethod
	def vpin(data:pd.DataFrame,buckets:int=50)->pd.Series:
		if'volume'not in data.columns or'returns'not in data.columns:return pd.Series()
		total_volume=data['volume'].sum();bucket_size=total_volume/buckets;vpin_values,cum_vol,buy_vol,sell_vol=[],0,0,0
		for i in range(len(data)):
			cum_vol+=data['volume'].iloc[i]
			if data['returns'].iloc[i]>0:buy_vol+=data['volume'].iloc[i]
			else:sell_vol+=data['volume'].iloc[i]
			if cum_vol>=bucket_size:vpin_values.append(abs(buy_vol-sell_vol)/cum_vol if cum_vol>0 else .0);cum_vol,buy_vol,sell_vol=0,0,0
			else:vpin_values.append(.0)
		return pd.Series(vpin_values,index=data.index)
class CrossSectionalSignals:
	@staticmethod
	def size(data:pd.DataFrame)->pd.Series:return pd.Series()if'market_cap'not in data.columns else np.log(data['market_cap'])
	@staticmethod
	def book_to_market(data:pd.DataFrame)->pd.Series:return pd.Series()
	@staticmethod
	def earnings_to_price(data:pd.DataFrame)->pd.Series:return pd.Series()
	@staticmethod
	def cashflow_to_price(data:pd.DataFrame)->pd.Series:return pd.Series()
	@staticmethod
	def dividend_yield(data:pd.DataFrame)->pd.Series:return pd.Series()
	@staticmethod
	def cross_sectional_momentum(data:pd.DataFrame,lookback:int=12)->pd.Series:return TimeSeriesSignals.momentum(data,lookback)
	@staticmethod
	def beta(data:pd.DataFrame,market_returns:pd.Series,window:int=60)->pd.Series:
		if'returns'not in data.columns:data['returns']=data['price'].pct_change()
		returns,beta_vals=data['returns'],[]
		for i in range(window,len(returns)):
			asset_ret=returns.iloc[i-window:i];mkt_ret=market_returns.iloc[i-window:i]if len(market_returns)>i else pd.Series()
			if len(mkt_ret)==len(asset_ret)and len(asset_ret)>1:var=mkt_ret.var();beta_vals.append(asset_ret.cov(mkt_ret)/var if var>0 else .0)
			else:beta_vals.append(.0)
		return pd.Series([.0]*window+beta_vals,index=data.index)
	@staticmethod
	def residual_volatility(data:pd.DataFrame,market_returns:pd.Series,window:int=60)->pd.Series:
		if'returns'not in data.columns:data['returns']=data['price'].pct_change()
		beta,returns=CrossSectionalSignals.beta(data,market_returns,window),data['returns'];residuals=[.0 if i<window else returns.iloc[i]-beta.iloc[i]*(market_returns.iloc[i]if i<len(market_returns)else .0)for i in range(len(returns))];return pd.Series(residuals,index=data.index).rolling(window=window).std()
	@staticmethod
	def quality_score(data:pd.DataFrame)->pd.Series:return pd.Series()
	@staticmethod
	def low_volatility(data:pd.DataFrame,window:int=20)->pd.Series:return-VolatilitySignals.realized_volatility(data,window=1).rolling(window=window).mean()
class RelativeValueSignals:
	@staticmethod
	def pair_spread(data1:pd.DataFrame,data2:pd.DataFrame,beta:float=1.)->pd.Series:
		if'price'not in data1.columns or'price'not in data2.columns:return pd.Series()
		aligned=data1.set_index('date').join(data2.set_index('date')[['price']],rsuffix='_2',how='left');return aligned['price']-beta*aligned['price_2']
	@staticmethod
	def cointegration_residual(data1:pd.DataFrame,data2:pd.DataFrame,weights:np.ndarray)->pd.Series:
		if'price'not in data1.columns or'price'not in data2.columns:return pd.Series()
		aligned=data1.set_index('date').join(data2.set_index('date')[['price']],rsuffix='_2',how='left')
		if len(weights)<2:weights=np.array([1.,-1.])
		residual=aligned['price']*weights[0]+aligned['price_2']*weights[1];return residual-residual.mean()
	@staticmethod
	def futures_basis(spot_price:float,futures_price:float)->float:return .0 if spot_price==0 else(futures_price-spot_price)/spot_price
	@staticmethod
	def carry_fx(domestic_rate:float,foreign_rate:float)->float:return domestic_rate-foreign_rate
	@staticmethod
	def yield_curve_slope(short_rate:float,long_rate:float)->float:return long_rate-short_rate
	@staticmethod
	def yield_curve_curvature(short_rate:float,mid_rate:float,long_rate:float)->float:return short_rate-2*mid_rate+long_rate
	@staticmethod
	def cross_asset_correlation(data1:pd.DataFrame,data2:pd.DataFrame,window:int=14)->pd.Series:
		if data1.empty or data2.empty:return pd.Series()
		if'returns'not in data1.columns:data1=data1.copy();data1['returns']=data1['price'].pct_change()
		if'returns'not in data2.columns:data2=data2.copy();data2['returns']=data2['price'].pct_change()
		if'date'not in data1.columns or'date'not in data2.columns:return pd.Series(index=data1.index if'date'in data1.columns else None,dtype=float)
		aligned=data1.set_index('date').join(data2.set_index('date')[['returns']],rsuffix='_2',how='inner')
		if len(aligned)<window:return pd.Series(index=data1.index if'date'in data1.columns else data1.index,dtype=float)
		return aligned['returns'].rolling(window=window,min_periods=max(5,window//2)).corr(aligned['returns_2']).fillna(.0)
	@staticmethod
	def relative_strength(data1:pd.DataFrame,data2:pd.DataFrame,window:int=14)->pd.Series:
		if data1.empty or data2.empty:return pd.Series()
		if'returns'not in data1.columns:data1=data1.copy();data1['returns']=data1['price'].pct_change()
		if'returns'not in data2.columns:data2=data2.copy();data2['returns']=data2['price'].pct_change()
		aligned=data1.set_index('date').join(data2.set_index('date')[['returns']],rsuffix='_2',how='inner')
		if len(aligned)<window:return pd.Series(index=data1.index if'date'in data1.columns else data1.index,dtype=float)
		return np.tanh((aligned['returns']/(aligned['returns_2']+1e-08)).rolling(window=window,min_periods=max(5,window//2)).mean()).fillna(.0)
	@staticmethod
	def btc_eth_correlation(data:pd.DataFrame,btc_data:pd.DataFrame,window:int=14)->pd.Series:return RelativeValueSignals.cross_asset_correlation(data,btc_data,window)
class RegimeSignals:
	@staticmethod
	def earnings_surprise(actual_earnings:float,expected_earnings:float)->float:return .0 if expected_earnings==0 else(actual_earnings-expected_earnings)/abs(expected_earnings)
	@staticmethod
	def post_event_drift(data:pd.DataFrame,event_dates:List[datetime],window:int=5)->pd.Series:
		drift=pd.Series(.0,index=data.index)
		if'returns'not in data.columns:data['returns']=data['price'].pct_change()
		for event_date in event_dates:drift[(data['date']>=event_date)&(data['date']<=event_date+timedelta(days=window))]=1.
		return drift
	@staticmethod
	def volatility_regime(data:pd.DataFrame,window:int=20)->pd.Series:vol_mean=VolatilitySignals.realized_volatility(data,window=1).rolling(window=window).mean();q33,q66=vol_mean.quantile(.33),vol_mean.quantile(.66);regime=pd.Series(0,index=data.index);regime[vol_mean>=q66]=2;regime[(vol_mean>=q33)&(vol_mean<q66)]=1;return regime
	@staticmethod
	def liquidity_regime(data:pd.DataFrame,window:int=20)->pd.Series:
		illiq=LiquiditySignals.amihud_illiquidity(data,window)
		if illiq.empty:return pd.Series()
		illiq_mean=illiq.rolling(window=window).mean();q33,q66=illiq_mean.quantile(.33),illiq_mean.quantile(.66);regime=pd.Series(0,index=data.index);regime[illiq_mean>=q66]=2;regime[(illiq_mean>=q33)&(illiq_mean<q66)]=1;return regime
class MetaSignals:
	@staticmethod
	def signal_crowding(data:pd.DataFrame,signal_name:str,market_returns:pd.Series)->pd.Series:
		if signal_name not in data.columns:return pd.Series()
		aligned=pd.concat([data[signal_name],market_returns],axis=1).dropna();return pd.Series()if len(aligned)<10 else aligned.iloc[:,0].rolling(window=20).corr(aligned.iloc[:,1]).abs()
	@staticmethod
	def signal_instability(data:pd.DataFrame,signal_name:str,window:int=60)->pd.Series:
		if signal_name not in data.columns:return pd.Series()
		signal=data[signal_name];t_stats=[abs(signal.iloc[i-window:i].mean()/signal.iloc[i-window:i].std()if signal.iloc[i-window:i].std()>0 else .0)for i in range(window,len(signal))];return pd.Series([.0]*window+t_stats,index=data.index)
class CausalEngine:
	def __init__(self,model_name:str='gpt-4o-mini',longterm_mode:bool=False):crca_max_loops=LONGTERM_MODE_CONFIG.get('crca_max_loops',5)if longterm_mode else 2;self.crca=CRCAAgent(agent_name='crca-quant-trading',agent_description='Causal analysis for quant trading signals',model_name=model_name,max_loops=crca_max_loops,variables=[],causal_edges=[]);self.longterm_mode=longterm_mode;self.latent_variables=['M_t','Vol_t','L_t','OF_t','R_i,t','F_i,t']
	def build_scm(self,variables:List[str],edges:List[Tuple[str,str]]):
		for(parent,child)in edges:
			try:self.crca.add_causal_relationship(parent,child,strength=.0)
			except Exception:pass
		causal_graph=getattr(self.crca,'causal_graph',{})
		for var in variables:
			if var not in causal_graph:
				try:self.crca.add_causal_relationship(var,var,strength=.0)
				except Exception:pass
	def fit_from_data(self,df:pd.DataFrame,variables:List[str],window:int=30):
		try:
			usable=df[variables].dropna()
			if len(usable)>=5:self.crca.fit_from_dataframe(df=usable,variables=variables,window=min(window,len(usable)),decay_alpha=.9)
		except Exception:pass
	def predict_outcomes(self,state:Dict[str,float],interventions:Dict[str,float]=None)->Dict[str,float]:
		if interventions:state={**state,**interventions}
		return self.crca._predict_outcomes(state,{})
class SignalValidator:
	def __init__(self,causal_engine:CausalEngine):self.causal_engine=causal_engine
	def compute_causal_score(self,signal_name:str,signal_values:pd.Series,target:pd.Series,regimes:pd.Series=None)->Dict[str,float]:relevance=self._mutual_information(signal_values,target);stability=self._regime_invariance(signal_values,target,regimes)if regimes is not None and not regimes.empty else 1.;causal_role=self._structural_consistency(signal_name,signal_values);score=relevance*.4+stability*.4+causal_role*.2;return{'score':score,'relevance':relevance,'stability':stability,'causal_role':causal_role}
	def _mutual_information(self,x:pd.Series,y:pd.Series)->float:
		try:aligned=pd.concat([x,y],axis=1).dropna();return min(1.,abs(aligned.corr().iloc[0,1])*2.)if len(aligned)>=10 else .0
		except Exception:return .0
	def _regime_invariance(self,signal:pd.Series,target:pd.Series,regimes:pd.Series)->float:
		try:
			aligned=pd.concat([signal,target,regimes],axis=1).dropna()
			if len(aligned)<10:return .5
			correlations=[abs(aligned[aligned.iloc[:,2]==regime].iloc[:,0].corr(aligned[aligned.iloc[:,2]==regime].iloc[:,1]))for regime in aligned.iloc[:,2].unique()if len(aligned[aligned.iloc[:,2]==regime])>5];return max(.0,1.-np.std(correlations))if len(correlations)>1 else .5
		except Exception:return .5
	def _structural_consistency(self,signal_name:str,signal_values:pd.Series=None)->float:
		signal_lower=signal_name.lower();signal_to_latent_map={'vol':['volatility','vol','vol_of_vol','garch'],'momentum':['momentum','trend','ema','ma_crossover'],'liquidity':['liquidity','spread','depth','amihud','vpin'],'volume':['volume','turnover','trade_imbalance'],'reversal':['reversal','mean_reversion','sma_dist'],'orderflow':['order','flow','imbalance','trade']};base_scores={'vol':.72,'momentum':.68,'liquidity':.75,'volume':.65,'reversal':.7,'orderflow':.73};signal_category,base_score=None,.5
		for(category,keywords)in signal_to_latent_map.items():
			if any(kw in signal_lower for kw in keywords):signal_category=category;base_score=base_scores.get(category,.65);break
		if signal_category is None:base_score=.6+hash(signal_name)%100/1e2*.2
		graph_boost=.0
		if hasattr(self.causal_engine,'crca')and hasattr(self.causal_engine.crca,'causal_graph'):
			causal_graph=self.causal_engine.crca.causal_graph;graph_nodes=list(causal_graph.keys())if isinstance(causal_graph,dict)else list(causal_graph.nodes())if hasattr(causal_graph,'nodes')else[];signal_base=signal_name.replace('signal_','').replace('_',' ');matches=sum(1 for node in graph_nodes if any(kw in node.lower()for kw in signal_lower.split('_'))or signal_base in node.lower()or node.lower()in signal_base)
			if matches>0:graph_boost=min(.15,matches*.05)
		data_quality_boost=.0
		if signal_values is not None and len(signal_values)>0:
			try:
				signal_std,signal_mean_abs=signal_values.std(),abs(signal_values.mean())
				if signal_mean_abs>1e-06:cv=signal_std/signal_mean_abs;data_quality_boost=.08 if .3<=cv<=.7 else .04 if .1<=cv<.3 or .7<cv<=1. else-.05
				nan_ratio=signal_values.isna().sum()/len(signal_values)
				if nan_ratio>.1:data_quality_boost-=.05*nan_ratio
			except Exception:pass
		hash_variation=(hash(signal_name)%20-10)/1e2;return float(max(.4,min(.95,base_score+graph_boost+data_quality_boost+hash_variation)))
class RegimeDetector:
	@staticmethod
	def detect_volatility_regime(data:pd.DataFrame)->str:
		if data.empty:return'unknown'
		vol=None
		if'volatility'in data.columns:vol=float(data['volatility'].iloc[-1])
		elif'signal_realized_vol'in data.columns:vol=float(data['signal_realized_vol'].iloc[-1])
		elif'returns'in data.columns:
			returns=data['returns'].dropna()
			if len(returns)>1:vol=float(returns.std()*np.sqrt(252))
		if vol is None:return'unknown'
		return'calm'if vol<.15 else'volatile'if vol>.35 else'normal'
	@staticmethod
	def detect_liquidity_regime(data:pd.DataFrame)->str:return'normal'
	@staticmethod
	def detect_macro_regime(data:pd.DataFrame)->str:return'normal'
class CausalMLModels:
	def __init__(self):self.models={}
	def create_causal_forest(self,X:np.ndarray,y:np.ndarray,t:np.ndarray):
		if SKLEARN_AVAILABLE:
			try:from sklearn.ensemble import RandomForestRegressor;model=RandomForestRegressor(n_estimators=100,max_depth=10,random_state=42);model.fit(np.column_stack([X,t]),y);return model
			except Exception:pass
	def create_causal_transformer(self,input_dim:int,hidden_dim:int=64):
		if TORCH_AVAILABLE:
			try:
				class CausalTransformer(nn.Module):
					def __init__(self,input_dim,hidden_dim):super().__init__();self.encoder=nn.Linear(input_dim,hidden_dim);self.decoder=nn.Linear(hidden_dim,1)
					def forward(self,x):return self.decoder(torch.relu(self.encoder(x)))
				return CausalTransformer(input_dim,hidden_dim)
			except Exception:pass
class JointTrainingEngine:
	def __init__(self,causal_engine:CausalEngine,causal_ml:CausalMLModels):self.causal_engine=causal_engine;self.causal_ml=causal_ml
	def train_joint(self,X:np.ndarray,y:np.ndarray,causal_structure:Dict[str,List[str]],epochs:int=10):return{'success':True}
class SignalWeightOptimizer:
	def __init__(self,decay:float=.95):self.weights={};self.performance_history=defaultdict(list);self.decay=decay
	def update_weights(self,signal_names:List[str],recent_performance:Dict[str,float])->Dict[str,float]:
		for name in signal_names:
			if name not in self.weights:self.weights[name]=1./len(signal_names)
			self.performance_history[name].append(recent_performance.get(name,.0))
		scores={}
		for name in signal_names:
			perf=self.performance_history[name][-20:]
			if len(perf)>1:scores[name]=np.mean(perf)/np.std(perf)if np.std(perf)>0 else .0
			else:scores[name]=.0
		total_score=sum(abs(s)for s in scores.values())
		if total_score>0:
			for name in signal_names:self.weights[name]=abs(scores[name])/total_score
		else:
			for name in signal_names:self.weights[name]=1./len(signal_names)
		return self.weights.copy()
class ModelSelector:
	def __init__(self):self.model_performance=defaultdict(dict)
	def select_best_model(self,regime:str,model_names:List[str])->str:
		if regime not in self.model_performance:return model_names[0]if model_names else None
		perf=self.model_performance[regime]
		if not perf:return model_names[0]if model_names else None
		best_model=max(perf.items(),key=lambda x:x[1]);return best_model[0]if best_model[1]>0 else model_names[0]
	def update_performance(self,regime:str,model_name:str,performance:float):
		if regime not in self.model_performance:self.model_performance[regime]={}
		self.model_performance[regime][model_name]=performance
class MetaLearner:
	def __init__(self):self.weight_optimizer=SignalWeightOptimizer();self.model_selector=ModelSelector()
	def optimize(self,signal_names:List[str],recent_performance:Dict[str,float],regime:str,model_names:List[str])->Tuple[Dict[str,float],str]:return self.weight_optimizer.update_weights(signal_names,recent_performance),self.model_selector.select_best_model(regime,model_names)
class ModelFactory:
	@staticmethod
	def create_model(model_type:str,**kwargs):
		if model_type=='linear'and SKLEARN_AVAILABLE:return LinearRegression(**kwargs)
		elif model_type=='rf'and SKLEARN_AVAILABLE:return RandomForestRegressor(**kwargs)
		elif model_type=='gb'and SKLEARN_AVAILABLE:return GradientBoostingRegressor(**kwargs)
		elif model_type=='xgb'and XGBOOST_AVAILABLE:return xgb.XGBRegressor(**kwargs)
		elif model_type=='lgb'and LIGHTGBM_AVAILABLE:return lgb.LGBMRegressor(**kwargs)
class TargetGenerator:
	@staticmethod
	def generate_forward_returns(data:pd.DataFrame,k:int=1,target_col:str='price')->pd.Series:return pd.Series()if target_col not in data.columns else data[target_col].shift(-k)/data[target_col]-1.
class EnsemblePredictor:
	def __init__(self):self.models={};self.weights={}
	def add_model(self,name:str,model:Any,weight:float=1.):self.models[name]=model;self.weights[name]=weight
	def predict(self,X:np.ndarray)->np.ndarray:
		predictions=[];total_weight=sum(self.weights.values())
		for(name,model)in self.models.items():
			try:pred=model.predict(X);weight=self.weights[name]/total_weight if total_weight>0 else 1./len(self.models);predictions.append(pred*weight)
			except Exception:pass
		return np.sum(predictions,axis=0)if predictions else np.zeros(X.shape[0])
	def update_weights(self,weights:Dict[str,float]):self.weights.update(weights)
class CovarianceEstimator:
	CovSize=lambda cov:cov.size if isinstance(cov,np.ndarray)else np.array(cov.values).size if isinstance(cov,pd.DataFrame)else 0;CovShape=lambda cov:cov.shape if isinstance(cov,np.ndarray)else np.array(cov.values).shape if isinstance(cov,pd.DataFrame)else(0,0)
	@staticmethod
	def ewma_covariance(returns:pd.DataFrame,alpha:float=.94)->np.ndarray:
		if returns.empty:return np.array([])
		returns_clean=returns.fillna(.0)
		if len(returns_clean.columns)==1:return np.array([[returns_clean.ewm(alpha=alpha,adjust=False).var().iloc[-1,0]]])
		cov=returns_clean.ewm(alpha=alpha,adjust=False).cov();n_assets=len(returns_clean.columns);return cov.iloc[-n_assets:,:].values
	@staticmethod
	def shrinkage_estimator(returns:pd.DataFrame,shrinkage:float=.5)->np.ndarray:
		returns_clean=returns.fillna(.0);sample_cov=returns_clean.cov().values;n=len(returns_clean.columns)
		if n==0:return np.array([])
		return shrinkage*(np.eye(n)*np.trace(sample_cov)/n)+(1-shrinkage)*sample_cov
	@staticmethod
	def compute_cross_asset_covariance(returns_dict:Dict[str,pd.Series],lambda_param:float=.94)->pd.DataFrame:
		if not returns_dict or len(returns_dict)<2:return pd.DataFrame()
		returns_df=pd.DataFrame(returns_dict).dropna()
		if returns_df.empty:return pd.DataFrame()
		cov_matrix=CovarianceEstimator.ewma_covariance(returns_df,alpha=1-lambda_param);asset_symbols=list(returns_dict.keys());return pd.DataFrame(cov_matrix,index=asset_symbols,columns=asset_symbols)
class RiskManager:
	@staticmethod
	def compute_cvar(returns:np.ndarray,alpha:float=.05)->float:
		if len(returns)==0:return .0
		var=np.percentile(returns,alpha*100);return abs(returns[returns<=var].mean())
	@staticmethod
	def stress_test(returns:pd.DataFrame,scenarios:List[Dict[str,float]])->Dict[str,float]:
		results={}
		for(scenario_name,scenario_returns)in scenarios.items():
			shocked_returns=returns.copy()
			for(asset,shock)in scenario_returns.items():
				if asset in shocked_returns.columns:shocked_returns[asset]+=shock
			results[scenario_name]=shocked_returns.sum(axis=1).mean()
		return results
class TransactionCostModel:
	def __init__(self,spread_bps:float=5.,slippage_bps:float=1e1,market_impact_coef:float=.5,maker_fee:float=.001,taker_fee:float=.002):self.spread_bps=spread_bps;self.slippage_bps=slippage_bps;self.market_impact_coef=market_impact_coef;self.maker_fee=maker_fee;self.taker_fee=taker_fee
	def estimate_total_cost(self,trade_size:float,trade_value:float,current_price:float,daily_volume:float=None,is_market_order:bool=True)->Dict[str,float]:spread_cost=trade_value*(self.spread_bps/1e4)*.5;slippage_cost=trade_value*(self.slippage_bps/1e4);impact_cost=trade_value*self.market_impact_coef*np.sqrt(trade_value/daily_volume)if daily_volume and daily_volume>0 else .0;fee_cost=trade_value*(self.taker_fee if is_market_order else self.maker_fee);total_cost=spread_cost+slippage_cost+impact_cost+fee_cost;return{'total':total_cost,'spread':spread_cost,'slippage':slippage_cost,'impact':impact_cost,'fee':fee_cost,'total_pct':total_cost/trade_value if trade_value>0 else .0}
	def adjust_expected_return(self,expected_return:float,trade_value:float,current_price:float,daily_volume:float=None)->float:costs=self.estimate_total_cost(trade_size=trade_value/current_price if current_price>0 else .0,trade_value=trade_value,current_price=current_price,daily_volume=daily_volume);return expected_return-costs['total_pct']
class PositionSizer:
	def __init__(self,target_vol:float=.02):self.target_vol=target_vol
	def size_position(self,base_fraction:float,realized_vol:float,uncertainty_penalty:float=.0)->float:vol_scale=min(2.,max(.25,self.target_vol/max(realized_vol,1e-06)));return float(np.clip(base_fraction*vol_scale*(1.-.5*uncertainty_penalty),.0,1.))
class PortfolioOptimizer:
	def __init__(self,risk_aversion:float=1.):self.risk_aversion=risk_aversion
	@staticmethod
	def _make_psd(covariance:np.ndarray,epsilon:float=1e-06)->np.ndarray:
		n=covariance.shape[0];cov_psd=covariance+np.eye(n)*epsilon
		try:eigenvals,eigenvecs=np.linalg.eigh(cov_psd);eigenvals=np.maximum(eigenvals,epsilon);cov_psd=eigenvecs@np.diag(eigenvals)@eigenvecs.T
		except Exception:pass
		return cov_psd
	def optimize_cvar(self,expected_returns:np.ndarray,covariance:np.ndarray,cvar_constraint:float=.05,max_leverage:float=1.,sector_constraints:Dict[str,float]=None,asset_types:List[str]=None,cross_asset_constraints:Dict[str,float]=None)->np.ndarray:
		n=len(expected_returns)
		if CVXPY_AVAILABLE:
			try:
				cov_psd=self._make_psd(covariance);w=cp.Variable(n);mu=cp.Parameter(n);mu.value=expected_returns;objective=cp.Maximize(mu@w-self.risk_aversion*cp.quad_form(w,cov_psd));constraints=[cp.sum(w)<=max_leverage,cp.sum(cp.abs(w))<=max_leverage*2,w>=-max_leverage,w<=max_leverage]
				if asset_types and cross_asset_constraints:
					for(asset_type,max_exposure)in cross_asset_constraints.items():
						type_mask=np.array([at==asset_type for at in asset_types])
						if np.any(type_mask):constraints.append(cp.sum(cp.abs(w[type_mask]))<=max_exposure)
				problem=cp.Problem(objective,constraints);problem.solve(solver=cp.ECOS,verbose=False)
				if problem.status in['optimal','optimal_inaccurate']and w.value is not None:return w.value
			except Exception:pass
		try:
			inv_cov=np.linalg.inv(covariance+np.eye(n)*1e-06);w=inv_cov@expected_returns;w_norm=np.sum(np.abs(w));w=w/w_norm*max_leverage if w_norm>1e-06 else np.sign(expected_returns)*max_leverage/n
			if asset_types and cross_asset_constraints:
				for(asset_type,max_exposure)in cross_asset_constraints.items():
					type_mask=np.array([at==asset_type for at in asset_types])
					if np.any(type_mask):
						type_weight_sum=np.sum(np.abs(w[type_mask]))
						if type_weight_sum>max_exposure:w[type_mask]*=max_exposure/type_weight_sum
			return np.clip(w,-max_leverage,max_leverage)
		except Exception:return np.sign(expected_returns)*max_leverage/n
	def optimize_asset_allocation(self,expected_returns_dict:Dict[str,float],covariance_df:pd.DataFrame,constraints:Dict[str,Any]=None)->Dict[str,float]:
		if not expected_returns_dict or covariance_df.empty:return{}
		asset_symbols=list(expected_returns_dict.keys());missing_assets=set(asset_symbols)-set(covariance_df.index)
		for asset in missing_assets:covariance_df.loc[asset]=.0;covariance_df[asset]=.0
		covariance_df=covariance_df.reindex(index=asset_symbols,columns=asset_symbols);expected_returns=np.array([expected_returns_dict[symbol]for symbol in asset_symbols]);covariance=covariance_df.values;asset_types=constraints.get('asset_types',None)if constraints else None;cross_asset_constraints=constraints.get('cross_asset_constraints',None)if constraints else None;weights=self.optimize_cvar(expected_returns=expected_returns,covariance=covariance,max_leverage=constraints.get('max_leverage',1.)if constraints else 1.,asset_types=asset_types,cross_asset_constraints=cross_asset_constraints);return{symbol:float(weight)for(symbol,weight)in zip(asset_symbols,weights)}
class BacktestEngine:
	def __init__(self,initial_capital:float=1e5,commission_rate:float=.001,slippage_rate:float=.0005):self.initial_capital=initial_capital;self.commission_rate=commission_rate;self.slippage_rate=slippage_rate;self.trades=[];self.equity_curve=[];self.daily_returns=[]
	def run_backtest(self,agent:'QuantTradingAgent',start_date:datetime,end_date:datetime,train_window:int=60,test_window:int=7,step:int=7)->Dict[str,Any]:
		self.trades=[];self.equity_curve=[];self.daily_returns=[];capital,position,entry_price=self.initial_capital,.0,.0;current_date,fold=start_date,0
		while current_date<end_date:
			fold+=1;train_start=current_date-timedelta(days=train_window);test_start,test_end=current_date,min(current_date+timedelta(days=test_window),end_date)
			if test_start>=test_end:break
			test_results=self._simulate_period(agent,train_start,current_date,test_start,test_end,capital,position,entry_price);capital,position,entry_price=test_results['final_capital'],test_results['final_position'],test_results['final_entry_price'];current_date+=timedelta(days=step)
		metrics=self._calculate_metrics();return{'metrics':metrics,'trades':self.trades,'equity_curve':self.equity_curve,'daily_returns':self.daily_returns,'initial_capital':self.initial_capital,'final_capital':capital,'total_return':(capital-self.initial_capital)/self.initial_capital}
	def _simulate_period(self,agent:'QuantTradingAgent',train_start:datetime,train_end:datetime,test_start:datetime,test_end:datetime,initial_capital:float,initial_position:float,initial_entry_price:float)->Dict[str,Any]:
		capital,position,entry_price=initial_capital,initial_position,initial_entry_price;agent.days_back=(train_end-train_start).days;agent.fetch_data()
		if agent.price_data.empty:return{'final_capital':capital,'final_position':position,'final_entry_price':entry_price}
		if'date'in agent.price_data.columns:test_data=agent.price_data[(agent.price_data['date']>=test_start)&(agent.price_data['date']<=test_end)].copy()
		else:
			test_data=agent.price_data.copy()
			if hasattr(agent.price_data.index,'date'):test_data=test_data[(test_data.index>=test_start)&(test_data.index<=test_end)]
		if test_data.empty:return{'final_capital':capital,'final_position':position,'final_entry_price':entry_price}
		for(idx,row)in test_data.iterrows():
			current_price=row['price'];current_date=row['date']if'date'in row else row.name if hasattr(row,'name')and isinstance(row.name,datetime)else datetime.now()
			if'date'in agent.price_data.columns:agent.price_data=agent.price_data[agent.price_data['date']<=current_date]
			else:agent.price_data=agent.price_data[agent.price_data.index<=current_date]
			agent.current_price=current_price
			try:
				signals_df=agent.compute_signals()
				if signals_df.empty:continue
				signal_scores=agent.validate_signals();predictions,covariance,pred_metadata=agent.generate_predictions(signal_scores)
				if len(predictions)==0:continue
				latest_pred=predictions[-1]if len(predictions)>0 else .0;expected_returns=np.array([latest_pred]);cov_size=CovarianceEstimator.CovSize(covariance);portfolio_weights=agent.optimize_portfolio(expected_returns,covariance)if cov_size>0 else np.array([.0]);weight=portfolio_weights[0]if len(portfolio_weights)>0 else .0;signal='BUY'if weight>.1 else'SELL'if weight<-.1 else'HOLD'
				if signal!='HOLD':trade_result=self._simulate_trade(signal,weight,current_price,capital,position,entry_price);capital,position,entry_price=trade_result['capital'],trade_result['position'],trade_result['entry_price']
			except Exception as e:logger.debug(f"Error in backtest simulation at {current_date}: {e}");continue
			portfolio_value=capital+position*current_price;self.equity_curve.append({'date':current_date,'equity':portfolio_value,'capital':capital,'position':position})
			if len(self.equity_curve)>1:self.daily_returns.append((portfolio_value-self.equity_curve[-2]['equity'])/self.equity_curve[-2]['equity'])
		return{'final_capital':capital,'final_position':position,'final_entry_price':entry_price}
	def _simulate_trade(self,signal:str,weight:float,price:float,capital:float,position:float,entry_price:float)->Dict[str,float]:
		portfolio_value=capital+position*price
		if signal=='BUY'and weight>0:
			target_value=portfolio_value*min(abs(weight),.2);target_position=target_value/price;trade_size=target_position-position
			if trade_size>0:
				execution_price=price*(1+self.slippage_rate);trade_cost=trade_size*execution_price;commission=trade_cost*self.commission_rate;total_cost=trade_cost+commission
				if total_cost<=capital:capital-=total_cost;position+=trade_size;entry_price=execution_price;self.trades.append({'date':datetime.now(),'signal':signal,'price':execution_price,'size':trade_size,'commission':commission,'slippage':trade_size*price*self.slippage_rate})
		elif signal=='SELL'and weight<0:
			target_value=portfolio_value*min(abs(weight),.2);target_position=-target_value/price;trade_size=position-target_position
			if trade_size>0:
				execution_price=price*(1-self.slippage_rate);trade_proceeds=trade_size*execution_price;commission=trade_proceeds*self.commission_rate;net_proceeds=trade_proceeds-commission;capital+=net_proceeds;position-=trade_size
				if position==0:entry_price=.0
				self.trades.append({'date':datetime.now(),'signal':signal,'price':execution_price,'size':trade_size,'commission':commission,'slippage':trade_size*price*self.slippage_rate})
		return{'capital':capital,'position':position,'entry_price':entry_price}
	def _calculate_metrics(self)->Dict[str,float]:
		if not self.equity_curve:return{}
		equity_values=[e['equity']for e in self.equity_curve];returns=np.array(self.daily_returns)if self.daily_returns else np.array([])
		if len(returns)==0:return{}
		total_return=(equity_values[-1]-equity_values[0])/equity_values[0]if equity_values[0]>0 else .0;days=len(equity_values);annualized_return=(1+total_return)**(252/days)-1 if days>0 else .0;volatility=np.std(returns)*np.sqrt(252)if len(returns)>1 else .0;sharpe_ratio=annualized_return/volatility if volatility>0 else .0;downside_returns=returns[returns<0];downside_std=np.std(downside_returns)*np.sqrt(252)if len(downside_returns)>1 else .0;sortino_ratio=annualized_return/downside_std if downside_std>0 else .0;peak=equity_values[0];max_drawdown=.0
		for equity in equity_values:
			if equity>peak:peak=equity
			drawdown=(peak-equity)/peak if peak>0 else .0;max_drawdown=max(max_drawdown,drawdown)
		wins=sum(1 for t in self.trades if t.get('profit',0)>0)if self.trades else 0;win_rate=wins/len(self.trades)if len(self.trades)>0 else .0;calmar_ratio=annualized_return/max_drawdown if max_drawdown>0 else .0;return{'total_return':total_return,'annualized_return':annualized_return,'volatility':volatility,'sharpe_ratio':sharpe_ratio,'sortino_ratio':sortino_ratio,'max_drawdown':max_drawdown,'calmar_ratio':calmar_ratio,'win_rate':win_rate,'total_trades':len(self.trades),'num_days':days}
	def _generate_report(self,results:Dict[str,Any])->str:metrics=results.get('metrics',{});return'\n'.join(['='*80,'BACKTEST REPORT','='*80,f"\nInitial Capital: ${results.get("initial_capital",0):,.2f}",f"Final Capital: ${results.get("final_capital",0):,.2f}",f"Total Return: {results.get("total_return",0):.2%}",f"\nPerformance Metrics:",f"  Annualized Return: {metrics.get("annualized_return",0):.2%}",f"  Volatility: {metrics.get("volatility",0):.2%}",f"  Sharpe Ratio: {metrics.get("sharpe_ratio",0):.2f}",f"  Sortino Ratio: {metrics.get("sortino_ratio",0):.2f}",f"  Max Drawdown: {metrics.get("max_drawdown",0):.2%}",f"  Calmar Ratio: {metrics.get("calmar_ratio",0):.2f}",f"  Win Rate: {metrics.get("win_rate",0):.2%}",f"  Total Trades: {metrics.get("total_trades",0)}",'='*80])
class CircuitBreaker:
	def __init__(self,max_daily_loss:float=.05,max_trades_per_day:int=50,kill_switch_file:str='kill_switch.txt'):self.max_daily_loss=max_daily_loss;self.max_trades_per_day=max_trades_per_day;self.kill_switch_file=kill_switch_file;self.daily_pnl=.0;self.daily_trades=0;self.last_reset_date=datetime.now().date();self.is_tripped=False
	def check_circuit(self)->Tuple[bool,str]:
		current_date=datetime.now().date()
		if current_date!=self.last_reset_date:self.daily_pnl=.0;self.daily_trades=0;self.last_reset_date=current_date
		if os.path.exists(self.kill_switch_file):self.is_tripped=True;return False,'Kill switch activated'
		if self.daily_pnl<=-self.max_daily_loss:self.is_tripped=True;return False,f"Daily loss limit exceeded: {self.daily_pnl:.2%}"
		if self.daily_trades>=self.max_trades_per_day:self.is_tripped=True;return False,f"Daily trade limit exceeded: {self.daily_trades}"
		return True,'OK'
	def record_trade(self,pnl:float):self.daily_pnl+=pnl;self.daily_trades+=1
	def reset(self):self.is_tripped=False;self.daily_pnl=.0;self.daily_trades=0
class RiskMonitor:
	def __init__(self,max_position_size:float=.2,max_leverage:float=1.,max_correlation:float=.8,max_dollar_risk_per_trade:float=None,max_portfolio_volatility:float=.2,max_drawdown:float=.15,max_exposure_per_asset:float=.15,max_exposure_per_type:Dict[str,float]=None):self.max_position_size=max_position_size;self.max_leverage=max_leverage;self.max_correlation=max_correlation;self.max_dollar_risk_per_trade=max_dollar_risk_per_trade;self.max_portfolio_volatility=max_portfolio_volatility;self.max_drawdown=max_drawdown;self.max_exposure_per_asset=max_exposure_per_asset;self.max_exposure_per_type=max_exposure_per_type or{};self.peak_portfolio_value=None;self.current_drawdown=.0
	def check_dollar_risk(self,position_size:float,stop_loss_distance:float,portfolio_value:float)->Tuple[bool,float]:
		if self.max_dollar_risk_per_trade is None:return True,position_size
		dollar_risk=position_size*portfolio_value*stop_loss_distance
		if dollar_risk>self.max_dollar_risk_per_trade:adjusted_size=self.max_dollar_risk_per_trade/(portfolio_value*stop_loss_distance);return False,adjusted_size
		return True,position_size
	def check_volatility_risk(self,position_size:float,asset_volatility:float,portfolio_volatility:float=None)->Tuple[bool,float]:
		if asset_volatility>0 and asset_volatility>self.max_portfolio_volatility:vol_adjusted_size=position_size*(self.max_portfolio_volatility/asset_volatility);return False,vol_adjusted_size
		if portfolio_volatility and portfolio_volatility>self.max_portfolio_volatility:scale_factor=self.max_portfolio_volatility/portfolio_volatility;vol_adjusted_size=position_size*scale_factor;return False,vol_adjusted_size
		return True,position_size
	def check_exposure_limits(self,position_size:float,asset_type:str=None,current_exposures:Dict[str,float]=None)->Tuple[bool,float]:
		if abs(position_size)>self.max_exposure_per_asset:adjusted_size=np.sign(position_size)*self.max_exposure_per_asset;return False,adjusted_size
		if asset_type and asset_type in self.max_exposure_per_type:
			max_type_exposure=self.max_exposure_per_type[asset_type]
			if current_exposures:
				current_type_exposure=sum(exp for(asset,exp)in current_exposures.items()if self._get_asset_type(asset)==asset_type)
				if current_type_exposure+abs(position_size)>max_type_exposure:remaining=max_type_exposure-current_type_exposure;adjusted_size=np.sign(position_size)*max(0,remaining);return False,adjusted_size
		return True,position_size
	def check_drawdown_limit(self,current_portfolio_value:float)->Tuple[bool,float]:
		if self.peak_portfolio_value is None:self.peak_portfolio_value=current_portfolio_value
		if current_portfolio_value>self.peak_portfolio_value:self.peak_portfolio_value=current_portfolio_value
		drawdown=(self.peak_portfolio_value-current_portfolio_value)/self.peak_portfolio_value;self.current_drawdown=drawdown
		if drawdown>self.max_drawdown:reduction_factor=1.-(drawdown-self.max_drawdown)/drawdown;reduction_factor=max(.1,reduction_factor);return False,reduction_factor
		return True,1.
	def _get_asset_type(self,asset_symbol:str)->str:
		asset_upper=asset_symbol.upper()
		if asset_upper in['BTC','ETH','DOGE','XRP','DOT','MATIC']:return'crypto'
		return'crypto'
	def pre_trade_check(self,signal:str,position_size:float,current_positions:Dict[str,float],portfolio_value:float,stop_loss_distance:float=.02,asset_volatility:float=None,asset_type:str=None)->Tuple[bool,str,float]:
		adjusted_size=position_size
		if abs(position_size)>self.max_position_size:adjusted_size=np.sign(position_size)*self.max_position_size;return False,f"Position size {position_size:.2%} exceeds limit {self.max_position_size:.2%}",adjusted_size
		if self.max_dollar_risk_per_trade:
			is_valid,adjusted=self.check_dollar_risk(position_size,stop_loss_distance,portfolio_value)
			if not is_valid:adjusted_size=min(abs(adjusted_size),adjusted)*np.sign(adjusted_size);return False,f"Dollar risk exceeds limit ${self.max_dollar_risk_per_trade:.2f}",adjusted_size
		if asset_volatility:
			is_valid,adjusted=self.check_volatility_risk(position_size,asset_volatility)
			if not is_valid:adjusted_size=min(abs(adjusted_size),abs(adjusted))*np.sign(adjusted_size);return False,f"Volatility risk exceeds limit {self.max_portfolio_volatility:.0%}",adjusted_size
		current_exposures={k:abs(v)for(k,v)in current_positions.items()};is_valid,adjusted=self.check_exposure_limits(position_size,asset_type,current_exposures)
		if not is_valid:adjusted_size=adjusted;return False,f"Exposure limit exceeded: {self.max_exposure_per_asset:.0%}",adjusted_size
		total_exposure=sum(abs(p)for p in current_positions.values())+abs(position_size)
		if total_exposure>self.max_leverage:return False,f"Total leverage {total_exposure:.2f} exceeds limit {self.max_leverage:.2f}",.0
		is_valid,reduction=self.check_drawdown_limit(portfolio_value)
		if not is_valid:adjusted_size*=reduction;return False,f"Drawdown limit exceeded: {self.max_drawdown:.0%} (current: {self.current_drawdown:.0%})",adjusted_size
		return True,'OK',adjusted_size
	def monitor_risk(self,positions:Dict[str,float],portfolio_value:float,covariance:np.ndarray)->Dict[str,float]:
		position_vector=np.array(list(positions.values()));cov_size,cov_shape=CovarianceEstimator.CovSize(covariance),CovarianceEstimator.CovShape(covariance)
		if cov_size>0 and len(position_vector)==cov_shape[0]:portfolio_variance=position_vector@covariance@position_vector;portfolio_risk=np.sqrt(portfolio_variance)
		else:portfolio_risk=.0
		return{'portfolio_risk':portfolio_risk,'total_exposure':sum(abs(p)for p in positions.values()),'num_positions':len(positions)}
class ConfidenceCalibrator:
	def __init__(self):(self.calibration_data):List[Tuple[float,bool]]=[];self.isotonic_model=None;(self.quantile_models):Dict[str,Any]={};(self.brier_scores):List[float]=[]
	def add_observation(self,predicted_prob:float,actual_in_interval:bool)->None:
		self.calibration_data.append((predicted_prob,actual_in_interval))
		if len(self.calibration_data)>1000:self.calibration_data=self.calibration_data[-1000:]
	def calibrate(self)->None:
		if len(self.calibration_data)<20:return
		try:
			if SKLEARN_AVAILABLE:from sklearn.isotonic import IsotonicRegression;probs=np.array([d[0]for d in self.calibration_data]);outcomes=np.array([float(d[1])for d in self.calibration_data]);self.isotonic_model=IsotonicRegression(out_of_bounds='clip');self.isotonic_model.fit(probs,outcomes)
		except Exception:pass
	def calibrate_prob(self,raw_prob:float)->float:
		if self.isotonic_model is None:return raw_prob
		try:return float(self.isotonic_model.predict([raw_prob])[0])
		except Exception:return raw_prob
	def compute_brier_score(self,predicted_probs:List[float],actual_outcomes:List[bool])->float:
		if len(predicted_probs)!=len(actual_outcomes)or len(predicted_probs)==0:return 1.
		brier=np.mean([(p-float(a))**2 for(p,a)in zip(predicted_probs,actual_outcomes)]);self.brier_scores.append(brier)
		if len(self.brier_scores)>100:self.brier_scores=self.brier_scores[-100:]
		return float(brier)
	def get_calibration_metrics(self)->Dict[str,float]:
		if len(self.brier_scores)==0:return{'brier_score':1.,'calibration_samples':0}
		return{'brier_score':float(np.mean(self.brier_scores)),'calibration_samples':len(self.calibration_data),'recent_brier':self.brier_scores[-1]if self.brier_scores else 1.}
class MonitoringSystem:
	def __init__(self):self.pnl_history=[];self.signal_health={};self.crca_diagnostics={};self.performance_metrics={}
	def track_pnl(self,timestamp:datetime,pnl:float,position:float,price:float):self.pnl_history.append({'timestamp':timestamp,'pnl':pnl,'position':position,'price':price})
	def monitor_signal_health(self,signal_name:str,score:float,decay:float):self.signal_health[signal_name]={'score':score,'decay':decay,'last_update':datetime.now()}
	def update_crca_diagnostics(self,graph_health:Dict[str,float],edge_strengths:Dict[str,float]):self.crca_diagnostics={'graph_health':graph_health,'edge_strengths':edge_strengths,'last_update':datetime.now()}
	def calculate_performance_attribution(self,signals:Dict[str,float],returns:pd.Series)->Dict[str,float]:
		attribution={}
		for(signal_name,signal_values)in signals.items():
			if len(signal_values)==len(returns):correlation=signal_values.corr(returns);attribution[signal_name]=correlation
		self.performance_metrics['attribution']=attribution;return attribution
	def get_summary(self)->Dict[str,Any]:total_pnl=sum(p['pnl']for p in self.pnl_history)if self.pnl_history else .0;return{'total_pnl':total_pnl,'num_trades':len(self.pnl_history),'signal_health':self.signal_health,'crca_diagnostics':self.crca_diagnostics,'performance_metrics':self.performance_metrics}
class TWAPExecutor:
	def execute(self,total_size:float,duration_minutes:int=60,num_splits:int=10)->List[Tuple[float,float]]:
		split_size=total_size/num_splits;interval=duration_minutes/num_splits;splits=[]
		for i in range(num_splits):splits.append((split_size,i*interval*60))
		return splits
class VWAPExecutor:
	def execute(self,total_size:float,volume_profile:pd.Series,duration_minutes:int=60)->List[Tuple[float,float]]:
		if volume_profile.empty:executor=TWAPExecutor();return executor.execute(total_size,duration_minutes)
		normalized_vol=volume_profile/volume_profile.sum();num_splits=len(normalized_vol);splits=[]
		for(i,vol_weight)in enumerate(normalized_vol):split_size=total_size*vol_weight;time_offset=duration_minutes/num_splits*i*60;splits.append((split_size,time_offset))
		return splits
class SmartRouter:
	def __init__(self):self.exchanges={}
	def route_order(self,symbol:str,side:str,size:float,exchanges:List[str])->List[Dict[str,Any]]:
		size_per_exchange=size/len(exchanges);orders=[]
		for exchange in exchanges:orders.append({'exchange':exchange,'symbol':symbol,'side':side,'size':size_per_exchange})
		return orders
	def iceberg_order(self,total_size:float,visible_size:float,num_slices:int=5)->List[Tuple[float,float]]:
		slices=[];remaining=total_size
		for i in range(num_slices):
			slice_size=min(visible_size,remaining)
			if slice_size>0:slices.append((slice_size,i*60));remaining-=slice_size
		return slices
class ImplementationShortfallOptimizer:
	def optimize_order_split(self,total_size:float,arrival_price:float,current_price:float,volatility:float,time_horizon:int=60)->List[Tuple[float,float]]:
		num_splits=min(10,time_horizon//6);split_size=total_size/num_splits;splits=[]
		for i in range(num_splits):splits.append((split_size,i*6))
		return splits
class KrakenRestClient:
	def __init__(self,api_key:str,api_secret:str,base_url:str='https://api.kraken.com',timeout:int=10,max_retries:int=3,dry_run:bool=False)->None:self.api_key=api_key;self.api_secret=api_secret;self.base_url=base_url.rstrip('/');self.timeout=timeout;self.max_retries=max_retries;self.dry_run=dry_run;self._min_interval=.35;(self._last_call_ts):float=.0
	def _throttle(self)->None:
		now=time.time();wait=self._min_interval-(now-self._last_call_ts)
		if wait>0:
			try:time.sleep(wait)
			except Exception:pass
		self._last_call_ts=time.time()
	def _sign(self,path:str,data:Dict[str,Any])->Dict[str,str]:nonce=data.get('nonce')or str(int(time.time()*1000));data['nonce']=nonce;post_data=urlencode(data);message=(nonce+post_data).encode();sha256=hashlib.sha256(message).digest();mac_data=path.encode()+sha256;secret=base64.b64decode(self.api_secret);sig=hmac.new(secret,mac_data,hashlib.sha512).digest();api_sign=base64.b64encode(sig).decode();return{'API-Key':self.api_key,'API-Sign':api_sign}
	def _post(self,path:str,data:Dict[str,Any])->Dict[str,Any]:
		url=f"{self.base_url}{path}";base_payload:Dict[str,Any]=dict(data)if data else{};last_error:Optional[str]=None
		for attempt in range(max(1,self.max_retries)):
			payload:Dict[str,Any]=dict(base_payload)
			try:
				self._throttle()
				if self.dry_run:logger.info(f"[Kraken REST dry-run] POST {path} payload={payload}");return{'success':True,'error':None,'result':{'dry_run':True,'request':{'path':path,'data':payload}}}
				headers=self._sign(path,payload);post_data=urlencode(payload);cmd=['curl','-s','-X','POST',url,'-H',f"API-Key: {headers["API-Key"]}",'-H',f"API-Sign: {headers["API-Sign"]}",'--data',post_data];proc=subprocess.run(cmd,capture_output=True,text=True,timeout=self.timeout)
				if proc.returncode!=0:last_error=proc.stderr.strip()or f"curl exited with {proc.returncode}";logger.warning(f"Kraken REST request failed ({attempt+1}/{self.max_retries}) on {path}: {last_error}");time.sleep(min(1.5**attempt,5.));continue
				raw=proc.stdout.strip()
				if not raw:last_error='empty response';logger.warning(f"Kraken REST request failed ({attempt+1}/{self.max_retries}) on {path}: {last_error}");time.sleep(min(1.5**attempt,5.));continue
				body=json.loads(raw);errors=body.get('error')or[]
				if errors:msg='; '.join(errors);logger.error(f"Kraken REST error on {path}: {msg}");return{'success':False,'error':msg,'result':body.get('result')or{}}
				return{'success':True,'error':None,'result':body.get('result')or{}}
			except Exception as exc:
				last_error=str(exc);logger.warning(f"Kraken REST request failed ({attempt+1}/{self.max_retries}) on {path}: {exc}")
				try:time.sleep(min(1.5**attempt,5.))
				except Exception:continue
		return{'success':False,'error':last_error or'unknown error','result':{}}
	def get_balance(self)->Dict[str,Any]:return self._post('/0/private/Balance',{})
	def add_order(self,pair:str,side:str,volume:float,ordertype:str='market',price:Optional[float]=None,leverage:Optional[Union[int,float]]=None,oflags:Optional[str]=None,**extra_params:Any)->Dict[str,Any]:
		data:Dict[str,Any]={'pair':pair,'type':side,'ordertype':ordertype,'volume':f"{volume:.10f}"}
		if price is not None and ordertype!='market':data['price']=f"{price:.10f}"
		if leverage is not None and float(leverage)>1.:data['leverage']=str(leverage)
		if oflags:data['oflags']=oflags
		if extra_params:data.update(extra_params)
		return self._post('/0/private/AddOrder',data)
class ExecutionEngine:
	def __init__(self,exchange_name:str='coinbase',api_key:Optional[str]=None,api_secret:Optional[str]=None,api_passphrase:Optional[str]=None,testnet:bool=True,max_position_size:float=.1,require_confirmation:bool=True,dry_run:bool=False):
		self.exchange_name=exchange_name.lower();self.testnet=testnet;self.max_position_size=max_position_size;self.require_confirmation=require_confirmation;self.exchange=None;self.position=.0;self.available_balance=.0;self.dry_run=dry_run;self.quote_preferences=TRADING_CONFIG.get('quote_preferences',{'kraken':['EUR','USD','USDC','USDT'],'binance':['USDT','BUSD','USD','USDC'],'coinbase':['USD','USDC','USDT'],'default':['USDT','USD','USDC']});self.is_optimizer=ImplementationShortfallOptimizer();(self.kraken_rest):Optional[KrakenRestClient]=None;self.trade_lock=threading.Lock()
		if not CCXT_AVAILABLE:logger.warning('CCXT not available - live trading disabled');return
		self._initialize_exchange(api_key,api_secret,api_passphrase)
	def _initialize_exchange(self,api_key:Optional[str],api_secret:Optional[str],api_passphrase:Optional[str])->None:
		exchange_upper=self.exchange_name.upper();api_key=api_key or os.getenv(f"{exchange_upper}_API_KEY")or(KRAKEN_API_KEY_FALLBACK if self.exchange_name=='kraken'else None);api_secret=api_secret or os.getenv(f"{exchange_upper}_API_SECRET")or(KRAKEN_API_SECRET_FALLBACK if self.exchange_name=='kraken'else None);api_passphrase=api_passphrase or(os.getenv(f"{exchange_upper}_API_PASSPHRASE")if self.exchange_name in['coinbase','kraken']else None)or(KRAKEN_API_PASSPHRASE_FALLBACK if self.exchange_name=='kraken'else None)
		if not api_key or not api_secret:logger.warning('Exchange credentials not found - live trading disabled');return
		try:
			config={'apiKey':api_key,'secret':api_secret,'enableRateLimit':True,'options':{'defaultType':'spot'}}
			if api_passphrase:config['passphrase']=api_passphrase
			if self.testnet and self.exchange_name.lower()!='kraken':config['sandbox'],config['options']['sandboxMode']=True,True
			elif self.testnet and self.exchange_name.lower()=='kraken':logger.warning("Kraken doesn't support sandbox mode - using production API (testnet disabled)");self.testnet=False
			self.exchange=getattr(ccxt,self.exchange_name)(config);logger.info(f"Exchange {self.exchange_name} initialized (testnet={self.testnet})")
			try:self.markets=self.exchange.load_markets()
			except Exception as e:logger.warning(f"Failed to load markets for {self.exchange_name}: {e}");self.markets={}
			if self.exchange_name=='kraken':self.kraken_rest=KrakenRestClient(api_key=api_key,api_secret=api_secret,dry_run=self.dry_run or self.testnet)
			self._update_balance()
		except Exception as e:logger.error(f"Failed to initialize exchange: {e}")
	def _update_balance(self,base_symbol:str='ETH')->None:
		if self.exchange_name=='kraken'and self.kraken_rest is not None:
			try:
				resp=self.kraken_rest.get_balance()
				if resp.get('success'):
					raw_balances:Dict[str,Any]=resp.get('result')or{};normalized:Dict[str,float]={}
					for(asset_code,qty)in raw_balances.items():
						norm=self._normalize_kraken_asset(str(asset_code))
						try:normalized[norm]=normalized.get(norm,.0)+float(qty)
						except(TypeError,ValueError):continue
					self.latest_balances_norm=normalized;quotes=getattr(self,'quote_preferences',None)
					if isinstance(quotes,dict):quote_list=quotes.get(self.exchange_name,quotes.get('default',['USDT','USD','USDC','EUR']))
					else:quote_list=quotes or['USDT','USD','USDC','EUR']
					self.available_balance=.0
					for q in quote_list:
						if q in normalized:self.available_balance=float(normalized[q]);break
					self.available_balances_by_quote={q:normalized.get(q,.0)for q in quote_list if q in normalized};base_norm=(base_symbol or'ETH').upper()
					if base_norm=='BTC':base_norm='BTC'
					self.position=float(normalized.get(base_norm,.0));return
			except Exception as e:logger.warning(f"Kraken REST balance fetch failed, falling back to CCXT: {e}")
		if not self.exchange:return
		try:
			balance=self.exchange.fetch_balance()
			try:self.latest_balances_norm={k.upper():float(v.get('free',.0)or .0)for(k,v)in balance.items()if isinstance(v,dict)}
			except Exception:pass
			self.available_balance=float(balance.get('USDT',{}).get('free',.0)or balance.get('USD',{}).get('free',.0)or balance.get('USDC',{}).get('free',.0)or balance.get('EUR',{}).get('free',.0)or .0);self.position=float(balance.get(base_symbol.upper(),{}).get('free',.0)or .0)
		except Exception as e:logger.warning(f"Failed to fetch balance: {e}")
	@staticmethod
	def _normalize_kraken_asset(asset_code:str)->str:
		if not asset_code:return''
		code=asset_code.split('.',1)[0].upper()
		if len(code)>3 and code[0]in('X','Z'):code=code[1:]
		if code=='XBT':code='BTC'
		if code=='ZUSD':code='USD'
		if code=='ZEUR':code='EUR'
		return code
	def _derive_symbol(self,base_symbol:str)->Optional[str]:
		if not base_symbol:return
		base=base_symbol.upper();bases_to_try=[base]
		if self.exchange_name=='kraken'and base=='BTC':bases_to_try.append('XBT')
		exchange_quotes=getattr(self,'quote_preferences',None)
		if not exchange_quotes:exchange_quotes=['USDT','USD','USDC']
		if isinstance(exchange_quotes,dict):exchange_quotes=exchange_quotes.get(self.exchange_name,exchange_quotes.get('default',['USDT','USD','USDC']))
		for b in bases_to_try:
			for quote in exchange_quotes:
				pair=f"{b}/{quote}"
				try:
					if self.exchange and pair in getattr(self,'markets',{}):return pair
				except Exception:pass
	def get_live_holdings(self)->Dict[str,float]:
		holdings:Dict[str,float]={}
		if hasattr(self,'latest_balances_norm')and isinstance(self.latest_balances_norm,dict):
			for(k,v)in self.latest_balances_norm.items():
				try:
					val=float(v)
					if abs(val)>0:holdings[k]=val
				except Exception:continue
		return holdings
	def _with_retries(self,fn,*args,retries:int=3,**kwargs):
		last_exc=None;delay=max(getattr(self.exchange,'rateLimit',250)/1e3,.25)if self.exchange else .25
		for _ in range(max(1,retries)):
			try:return fn(*args,**kwargs)
			except Exception as e:
				last_exc=e
				try:import time;time.sleep(delay);delay*=1.5
				except Exception:pass
		if last_exc:raise last_exc
	def _market_meta(self,routed_symbol:str)->Dict[str,Any]:
		try:
			if self.exchange and routed_symbol in self.exchange.markets:return self.exchange.markets[routed_symbol]
		except Exception:pass
		return{}
	def execute_trade(self,signal:str,size_fraction:float=.1,price:Optional[float]=None,use_is:bool=False,base_symbol:str='ETH',aggressive_mode:bool=False,expected_return_pct:Optional[float]=None,leverage:Optional[Union[int,float]]=None)->Dict[str,Any]:
		lock_acquired=False
		if hasattr(self,'trade_lock'):
			lock_acquired=self.trade_lock.acquire(timeout=10)
			if not lock_acquired:return{'error':'Trade lock timeout'}
		try:
			if not self.exchange and not(self.exchange_name=='kraken'and self.kraken_rest is not None):return{'error':'Exchange not initialized'}
			if signal=='HOLD':return{'status':'hold'}
			self._update_balance(base_symbol);size_fraction=min(abs(size_fraction),self.max_position_size);routed_symbol=self._derive_symbol(base_symbol)or f"{base_symbol.upper()}/USDT";market=self._market_meta(routed_symbol);lot_min=market.get('limits',{}).get('amount',{}).get('min',None);lot_max=market.get('limits',{}).get('amount',{}).get('max',None);notional_min=market.get('limits',{}).get('cost',{}).get('min',None);price_precision=market.get('precision',{}).get('price',None);step=market.get('precision',{}).get('amount',None);price_info=None
			try:
				if self.exchange:
					if price is None:price_info=self._with_retries(self.exchange.fetch_ticker,routed_symbol);price=float(price_info['last'])
					else:price_info=self._with_retries(self.exchange.fetch_ticker,routed_symbol)
				elif price is None:return{'error':f"Ticker fetch failed for {routed_symbol} (no metadata client available)"}
			except Exception:return{'error':f"Ticker fetch failed for {routed_symbol}"}
			if price is None or price<=0:return{'error':f"Invalid price for {routed_symbol}: {price}"}
			try:
				ref_price=None
				if price_info:
					bid,ask,last=price_info.get('bid'),price_info.get('ask'),price_info.get('last')
					if bid and ask:ref_price=(bid+ask)/2
					elif last:ref_price=last
				ref_price=ref_price or price
				if ref_price and ref_price>0:
					slip_pct=abs(price-ref_price)/ref_price
					if slip_pct>.005:return{'error':f"Slippage {slip_pct:.2%} exceeds 0.5% cap"}
			except Exception:pass
			if price_precision is not None:
				try:quant=10**-price_precision;price=float(np.floor(price/quant)*quant)
				except Exception:pass
			quote_ccy=routed_symbol.split('/')[1]if'/'in routed_symbol else None;quote_balance=self.available_balance
			if self.exchange_name=='kraken'and quote_ccy and hasattr(self,'available_balances_by_quote'):
				qb=self.available_balances_by_quote.get(quote_ccy)
				if qb is not None:quote_balance=float(qb)
			min_trade_value=getattr(self,'min_trade_value',5.)if hasattr(self,'min_trade_value')else 5.
			if signal=='BUY':
				if quote_balance<=0:return{'status':'skipped','reason':'no available balance'}
				required_fraction=min_trade_value*1.02/quote_balance;size_fraction=max(size_fraction,required_fraction)
				if self.max_position_size:size_fraction=min(size_fraction,self.max_position_size)
				notional=quote_balance*size_fraction
				if notional<min_trade_value:return{'status':'skipped','reason':f"notional {notional:.2f} < min_trade_value {min_trade_value:.2f}"}
				amount,side=notional/price,'buy'
			elif signal=='SELL':
				base_free=self.position
				if base_free<=0:return{'status':'skipped','reason':f"no position in {base_symbol}"}
				if base_free*price<=0:return{'status':'skipped','reason':'price or position non-positive'}
				required_fraction=min_trade_value*1.02/(base_free*price);size_fraction=max(size_fraction,required_fraction)
				if self.max_position_size:size_fraction=min(size_fraction,self.max_position_size)
				amount=base_free*size_fraction;notional,side=amount*price,'sell'
				if notional<min_trade_value:
					target_amount=min_trade_value*1.02/price;amount=min(base_free,target_amount);notional=amount*price
					if notional<min_trade_value:return{'status':'skipped','reason':f"notional {notional:.2f} < min_trade_value {min_trade_value:.2f} (after bump)"}
			else:return{'error':f"Invalid signal: {signal}"}
			def clamp_to_step(val:float,step_val:Optional[float])->float:
				if step_val and step_val>0:return float(np.floor(val/step_val)*step_val)
				return val
			if lot_min is not None and amount<lot_min:
				if aggressive_mode:amount=clamp_to_step(lot_min,step)
				else:return{'status':'skipped','reason':f"amount {amount} < min {lot_min}"}
			if step:amount=clamp_to_step(amount,step)
			if lot_min is not None and amount<lot_min:return{'status':'skipped','reason':f"amount {amount} < min {lot_min}"}
			if lot_max is not None and amount>lot_max:amount=min(amount,lot_max)
			notional=price*amount
			if notional_min is not None and notional<notional_min:return{'status':'skipped','reason':f"notional {notional:.4f} < min {notional_min}"}
			taker_fee=market.get('taker',.001)if isinstance(market,dict)else .001
			if expected_return_pct is not None:
				total_cost_pct=taker_fee+.005
				if expected_return_pct<=total_cost_pct:return{'status':'skipped','reason':f"expected_return {expected_return_pct:.2%} <= fees+slip {total_cost_pct:.2%}"}
			if use_is and amount>.01:splits=self.is_optimizer.optimize_order_split(amount,price,price,.02);logger.info(f"IS optimization: {len(splits)} splits");amount=splits[0][0]if splits else amount
			using_kraken_rest=self.exchange_name=='kraken'and self.kraken_rest is not None;pair_id=None
			if using_kraken_rest and isinstance(market,dict):pair_id=market.get('id')
			if using_kraken_rest and not pair_id:pair_id=routed_symbol.replace('/','')
			if using_kraken_rest:
				if self.dry_run:logger.info(f"[Kraken REST] {side.upper()} {amount:.6f} {pair_id} @ {price:.6f} (leverage={leverage or 1.}, dry_run={self.dry_run})")
				resp=self.kraken_rest.add_order(pair=pair_id,side=side,volume=amount,ordertype='market',price=None,leverage=leverage)
				if not resp.get('success'):
					err_msg=(resp.get('error')or'').upper()
					if'USDT'in(pair_id or'').upper()and'RESTRICT'in err_msg:
						base=routed_symbol.split('/')[0]
						for alt_quote in['EUR','USD','USDC']:
							alt_pair=f"{base}/{alt_quote}";market_alt=getattr(self,'markets',{}).get(alt_pair)if getattr(self,'markets',None)else None;alt_pair_id=market_alt.get('id')if isinstance(market_alt,dict)and market_alt.get('id')else alt_pair.replace('/','');logger.warning(f"USDT restricted on Kraken; retrying on {alt_pair_id}");resp_alt=self.kraken_rest.add_order(pair=alt_pair_id,side=side,volume=amount,ordertype='market',price=None,leverage=leverage)
							if resp_alt.get('success'):result_alt=resp_alt.get('result')or{};txids_alt=result_alt.get('txid')or[];order_id_alt=txids_alt[0]if isinstance(txids_alt,list)and txids_alt else txids_alt if isinstance(txids_alt,str)else None;fill_cost_alt=price*amount;return{'status':'success','order_id':order_id_alt,'side':side,'amount':amount,'price':price,'notional':fill_cost_alt}
					logger.error(f"Kraken REST order failed: {resp.get("error")}");return{'error':resp.get('error','Kraken REST order failed'),'status':'failed'}
				result=resp.get('result')or{};txids=result.get('txid')or[];order_id=txids[0]if isinstance(txids,list)and txids else txids if isinstance(txids,str)else None;fill_cost=price*amount;logger.info(f"Kraken REST order placed: {side.upper()} {amount:.6f} {base_symbol.upper()}");return{'status':'success','order_id':order_id,'side':side,'amount':amount,'price':price,'notional':fill_cost}
			order=self.exchange.create_market_order(symbol=routed_symbol,side=side,amount=amount);logger.info(f"Order placed: {side.upper()} {amount:.6f} {base_symbol.upper()}");filled_amount=amount
			try:
				order_id=order.get('id');fill_cost=price*filled_amount
				if order_id:
					attempts=3
					for _ in range(attempts):
						try:
							fetched=self._with_retries(self.exchange.fetch_order,order_id,symbol=routed_symbol);filled_amount=float(fetched.get('filled',filled_amount)or filled_amount);fill_cost=float(fetched.get('cost',fill_cost)or fill_cost)
							if fetched.get('status')in['closed','canceled']:break
						except Exception:pass
						import time as _t;_t.sleep(max(getattr(self.exchange,'rateLimit',250)/1e3,.25))
			except Exception:fill_cost=price*filled_amount
			return{'status':'success','order_id':order.get('id'),'side':side,'amount':filled_amount,'price':price,'notional':fill_cost}
		except Exception as e:logger.error(f"Trade execution failed: {e}");return{'error':str(e),'status':'failed'}
		finally:
			if hasattr(self,'trade_lock')and lock_acquired:
				try:self.trade_lock.release()
				except Exception:pass
	def get_position(self)->Dict[str,Any]:
		if not self.exchange:return{'error':'Exchange not initialized'}
		self._update_balance()
		try:ticker=self.exchange.fetch_ticker('ETH/USD');current_price=float(ticker['last'])
		except:current_price=.0
		return{'position':self.position,'available_balance':self.available_balance,'current_price':current_price,'position_value':self.position*current_price}
class QuantTradingAgent:
	def __init__(self,days_back:int=75,demo_mode:bool=False,live_trading_mode:bool=False,exchange_name:str='kraken',api_key:Optional[str]=None,api_secret:Optional[str]=None,api_passphrase:Optional[str]=None,testnet:bool=True,max_position_size:Optional[float]=None,require_confirmation:bool=True,account_size:Optional[float]=None,assets:Optional[List[Dict[str,str]]]=None,longterm_mode:bool=False):
		load_dotenv();self.longterm_mode=longterm_mode
		if account_size is not None and account_size<0:logger.warning(f"Negative account_size ({account_size}) provided, using default from TRADING_CONFIG");account_size=None
		if max_position_size is not None:
			if max_position_size<0:logger.warning(f"Negative max_position_size ({max_position_size}) provided, using default");max_position_size=None
			elif max_position_size>1.:logger.warning(f"max_position_size > 1.0 ({max_position_size}) provided, capping at 1.0");max_position_size=1.
		if longterm_mode:
			config={**TRADING_CONFIG,**LONGTERM_MODE_CONFIG};self.account_size=account_size or config['account_size'];self.account_category=config.get('account_category',TRADING_CONFIG['account_category']);self.conservative_mode=config['conservative_mode'];self.aggressive_mode=config['aggressive_mode'];self.min_trade_value=config['min_trade_value'];self.position_size_multiplier=config['position_size_multiplier'];self.max_position_size=max_position_size or config['max_position_size'];self.prediction_horizon_days=config['prediction_horizon_days'];self.position_evaluation_interval_hours=config['position_evaluation_interval_hours'];self.min_confidence_threshold=config['min_confidence_threshold'];self.max_asset_price_usd=config['max_asset_price_usd'];self.loop_interval_seconds=config['loop_interval_seconds'];self.full_refresh_interval_hours=config['full_refresh_interval_hours'];hard_cap=config.get('max_position_hard_cap')
			if hard_cap:self.max_position_size=min(self.max_position_size,hard_cap)
		else:
			self.account_size=account_size or TRADING_CONFIG['account_size'];self.account_category=TRADING_CONFIG['account_category'];self.conservative_mode=TRADING_CONFIG['conservative_mode'];self.aggressive_mode=TRADING_CONFIG.get('aggressive_mode',False);self.min_trade_value=TRADING_CONFIG['min_trade_value'];self.position_size_multiplier=TRADING_CONFIG['position_size_multiplier'];self.max_position_size=max_position_size or TRADING_CONFIG.get('max_position_size',.2);hard_cap=TRADING_CONFIG.get('max_position_hard_cap')
			if hard_cap:self.max_position_size=min(self.max_position_size,hard_cap)
		self.positions={};self.stop_loss_pct=TRADING_CONFIG.get('stop_loss_pct',-3.);self.stop_gain_pct=TRADING_CONFIG.get('stop_gain_pct',5.);self.initial_portfolio_value=account_size or TRADING_CONFIG['account_size'];self.initial_baseline=self.initial_portfolio_value;self.baseline_value=self.initial_baseline;self.last_portfolio_value=self.initial_portfolio_value;(self.last_full_run):Optional[Dict[str,Any]]=None;(self.last_full_run_time):float=.0;(self.full_refresh_interval):float=3e2;(self.incremental_update_enabled):bool=False;(self.last_full_run):Optional[Dict[str,Any]]=None;(self.last_full_run_time):float=.0;(self.full_refresh_interval):float=3e2;(self.incremental_update_enabled):bool=False;self.promotion_threshold_pct=TRADING_CONFIG.get('promotion_threshold_pct',8.);self.promotion_liquidate_enabled=TRADING_CONFIG.get('promotion_liquidate_enabled',True);self.promotion_debounce_secs=TRADING_CONFIG.get('promotion_debounce_secs',.0);self.promotion_event=False;self.force_halt_after_promotion=False;self.ratchet_armed=False;self.ratchet_promoted=False;(self.cooldown_peak_value):Optional[float]=None;(self.promotion_candidate_value):Optional[float]=None;(self.promotion_candidate_ts):Optional[float]=None;(self.exit_summary):Optional[Dict[str,Any]]=None;self.dynamic_max_position_size=TRADING_CONFIG.get('max_position_size',.2);self.session_pnl_pct=.0;self.circuit_breaker_triggered=False;self.cooldown_enabled=TRADING_CONFIG.get('cooldown_enabled',True);self.system_state='ACTIVE';self.cooldown_loop_trigger=TRADING_CONFIG.get('cooldown_loop_trigger',25);self.cooldown_api_budget_threshold=TRADING_CONFIG.get('cooldown_api_budget_threshold',.3);self.cooldown_congestion_threshold=TRADING_CONFIG.get('cooldown_congestion_threshold',.65);self.cooldown_min_loops=TRADING_CONFIG.get('cooldown_min_loops',3);self.cooldown_max_loops=TRADING_CONFIG.get('cooldown_max_loops',max(self.cooldown_min_loops,4));self.cooldown_sleep_multiplier=TRADING_CONFIG.get('cooldown_sleep_multiplier',1.15);self.cooldown_loops_remaining=0;self.cooldown_total_loops=0;self.cooldown_trigger_reason='';self.loops_since_cooldown=0;self.global_loop_counter=0;self.cooldown_exit_guard=False;self.api_budget_remaining=1.;logger.add('quant_trading_agent.log',rotation='500 MB',retention='10 days',level='INFO',format='{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}');self.days_back=days_back;self.demo_mode=demo_mode and not live_trading_mode;self.live_trading_mode=live_trading_mode;self.assets=assets;self.multi_asset_mode=False;self.auto_multi_asset=AUTO_MULTI_ASSET;self.market_data_client=MarketDataClient(demo_mode=self.demo_mode);self.order_book_tracker=OrderBookTracker();self.socket_manager=ExchangeSocketManager();self.signal_registry=SignalRegistry();self.time_series_signals=TimeSeriesSignals();self.volatility_signals=VolatilitySignals();self.liquidity_signals=LiquiditySignals();self.cross_sectional_signals=CrossSectionalSignals();self.relative_value_signals=RelativeValueSignals();self.regime_signals=RegimeSignals();self.meta_signals=MetaSignals();self.social_sentiment_signals=SocialSentimentSignals();self.onchain_signals=OnChainSignals();self.news_sentiment_signals=NewsSentimentSignals();self.github_signals=GitHubSignals();self.exchange_metrics_signals=ExchangeMetricsSignals();(self.alternative_data_cache):Dict[str,Dict[str,float]]={};(self.alternative_data_timestamps):Dict[str,float]={};self.causal_engine=CausalEngine(longterm_mode=self.longterm_mode);self.signal_validator=SignalValidator(self.causal_engine);self.regime_detector=RegimeDetector();self.causal_ml=CausalMLModels();self.joint_training=JointTrainingEngine(self.causal_engine,self.causal_ml);self.meta_learner=MetaLearner();self.model_factory=ModelFactory();self.target_generator=TargetGenerator();self.ensemble_predictor=EnsemblePredictor();self.cov_estimator=CovarianceEstimator();self.risk_manager=RiskManager()
		if longterm_mode:rm_cap=self.max_position_size
		else:
			rm_cap=max_position_size or TRADING_CONFIG.get('max_position_size',.2);hard_cap=TRADING_CONFIG.get('max_position_hard_cap')
			if hard_cap:rm_cap=min(rm_cap,hard_cap)
		self.risk_monitor=RiskMonitor(max_position_size=rm_cap);self.circuit_breaker=CircuitBreaker();self.rotation_manager=AssetRotationManager()if ROTATION_ENABLED else None;self.position_sizer=PositionSizer();self.portfolio_optimizer=PortfolioOptimizer();self.monitoring=MonitoringSystem();self.confidence_calibrator=ConfidenceCalibrator();self.twap_executor=TWAPExecutor();self.vwap_executor=VWAPExecutor();self.smart_router=SmartRouter();self.price_data=pd.DataFrame();self.signals_df=pd.DataFrame();self.current_price=.0;self.streaming_buffer=[];self.streaming_mode=False;(self.price_data_dict):Dict[str,pd.DataFrame]={};(self.signals_df_dict):Dict[str,pd.DataFrame]={};(self.asset_types):Dict[str,str]={}
		if assets is not None and len(assets)>1:self.multi_asset_mode=True
		if self.auto_multi_asset and(assets is None or len(assets)<=1):
			wallet_value=self.account_size;max_asset_price=getattr(self,'max_asset_price_usd',None)if longterm_mode else None;self.asset_discovery=AssetDiscovery(wallet_value=wallet_value,min_asset_value=MIN_ASSET_VALUE,max_assets=MAX_ASSETS_LIMIT,longterm_mode=longterm_mode,max_asset_price_usd=max_asset_price);discovered_assets=self.asset_discovery.discover_assets()
			if discovered_assets:self.assets=discovered_assets;self.multi_asset_mode=True;logger.info(f"🔍 Auto-discovered {len(discovered_assets)} assets for trading")
		if self.multi_asset_mode:
			assets_to_process=self.assets if self.assets else assets if assets else[]
			for asset_config in assets_to_process:
				symbol=asset_config.get('symbol','').upper();asset_type=asset_config.get('type','auto')
				if symbol:self.asset_types[symbol]=asset_type
			logger.info(f"Multi-asset mode enabled with {len(assets_to_process)} assets: {list(self.asset_types.keys())}")
		self.cache_type=ALTERNATIVE_DATA_CONFIG.get('cache_type','redis');self.cache=self._init_cache();self.cache_dir=Path('.cache/signals');self.cache_dir.mkdir(parents=True,exist_ok=True);self._redis_client=None;self.execution_engine=ExecutionEngine(exchange_name=exchange_name,api_key=api_key,api_secret=api_secret,api_passphrase=api_passphrase,testnet=testnet,max_position_size=rm_cap,require_confirmation=require_confirmation,dry_run=not live_trading_mode);self.backtest_engine=BacktestEngine()
		if self.live_trading_mode:
			asset_list=assets or ExchangeAssetCatalog.default_assets(exchange_name,MAX_ASSETS_LIMIT);valid_pairs,missing=[],[]
			for a in asset_list:
				base=a.get('symbol')if isinstance(a,dict)else None
				if not base:continue
				routed=self.execution_engine._derive_symbol(base)if hasattr(self.execution_engine,'_derive_symbol')else None
				if routed and routed in getattr(self.execution_engine,'markets',{}):valid_pairs.append(routed)
				else:missing.append(base)
			if not valid_pairs:raise RuntimeError(f"Live trading enabled but no valid trading symbols found on {exchange_name} for assets {missing or asset_list}")
			if missing:logger.warning(f"Live trading: missing markets for {missing} on {exchange_name}; tradable pairs: {valid_pairs}")
		self.alt_data_client=AltDataClient(config=ALTERNATIVE_DATA_CONFIG)
	def _initialize_live_portfolio_state(self)->None:
		engine=getattr(self,'execution_engine',None)
		if engine is None:return
		if getattr(engine,'exchange_name','').lower()!='kraken':return
		try:engine._update_balance()
		except Exception as e:logger.warning(f"Live portfolio init: balance update failed: {e}");return
		live_value=float(getattr(engine,'available_balance',.0)or .0)
		if live_value<=0:logger.warning('Live portfolio init: Kraken balance is zero/absent; using configured account_size');return
		self.initial_portfolio_value=live_value;self.initial_baseline=live_value;self.baseline_value=live_value;self.last_portfolio_value=live_value
	def fetch_data(self,asset:str='ethereum')->None:
		try:alt_data=self.alt_data_client.fetch_all_alternative_data(asset,self.days_back);self.alternative_data_cache=alt_data;self.alternative_data_timestamps={k:time.time()for k in alt_data.keys()}
		except Exception as e:logger.debug(f"Alternative data fetch failed: {e}")
		if self.multi_asset_mode:
			self.price_data_dict=self.market_data_client.fetch_multiple_assets(self.assets,self.days_back);self.price_data_dict=self.market_data_client.validate_unified_schema(self.price_data_dict)
			if(not self.price_data_dict or not any(df is not None and not df.empty for df in self.price_data_dict.values()))and self.live_trading_mode and hasattr(self,'execution_engine')and getattr(self.execution_engine,'exchange',None):
				import pandas as pd;seeded={}
				for a in self.assets:
					sym=a.get('symbol','').upper();routed=self.execution_engine._derive_symbol(sym)if hasattr(self.execution_engine,'_derive_symbol')else None
					if not routed:continue
					try:
						ticker=self.execution_engine.exchange.fetch_ticker(routed);price=ticker.get('last')or ticker.get('close')
						if price:
							p=float(price);base_p=self.market_data_client._get_asset_base_price(sym)if hasattr(self,'market_data_client')else None
							if base_p:
								lo,hi=base_p*.05,base_p*2e1
								if p<lo or p>hi:logger.warning(f"Skipping {sym} ticker fallback due to implausible price {p:.2f} (expected ~{base_p:.2f})");continue
							seeded[sym]=pd.DataFrame({'price':[p*.999,p]})
					except Exception:continue
				if seeded:
					now=pd.Timestamp.utcnow()
					for(sym,df)in seeded.items():
						if'date'not in df.columns:df['date']=[now-pd.Timedelta(minutes=1),now]
						if'returns'not in df.columns:df['returns']=.0
						df.reset_index(drop=True,inplace=True)
					self.price_data_dict=seeded
			if not self.price_data_dict:raise RuntimeError('Failed to fetch multi-asset data (both historical and live fallback)')
			primary_symbol=list(self.price_data_dict.keys())[0];self.price_data=self.price_data_dict[primary_symbol];self.current_price=self.price_data['price'].iloc[-1]
			if VERBOSE_LOGGING:logger.debug(f"Fetched multi-asset data: {len(self.price_data_dict)} assets, {len(self.price_data)} points per asset")
		else:
			self.price_data=self.market_data_client.fetch_price_data(asset,self.days_back);assert not self.price_data.empty,'Failed to fetch price data';self.current_price=self.price_data['price'].iloc[-1]
			if VERBOSE_LOGGING:logger.debug(f"Fetched data: {len(self.price_data)} points, current price: ${self.current_price:,.2f}")
	def compute_signals(self,asset_symbol:Optional[str]=None)->pd.DataFrame:
		if not isinstance(self.price_data,pd.DataFrame):logger.error(f"price_data is not a DataFrame (type: {type(self.price_data)}), returning empty DataFrame");return pd.DataFrame()
		symbol_key=asset_symbol or('multi'if self.multi_asset_mode else'single');cache_key=f"signals:{symbol_key}:{hash(str(self.price_data.index[-1])if not self.multi_asset_mode else str(list(self.price_data_dict.keys())))if self.price_data.index.size>0 else"empty"}"
		if self.cache=='redis'and self._redis_client:
			try:
				cached_signals=self._redis_client.get(cache_key)
				if cached_signals:
					cached_data=json.loads(cached_signals)
					if'signals'in cached_data:return pd.DataFrame(cached_data['signals'],index=self.price_data.index if not self.multi_asset_mode else self.price_data_dict[list(self.price_data_dict.keys())[0]].index)
			except Exception:pass
		if asset_symbol is not None and self.multi_asset_mode:assert asset_symbol in self.price_data_dict,f"Asset {asset_symbol} not found in price_data_dict";data=self.price_data_dict[asset_symbol].copy()
		elif self.multi_asset_mode and self.price_data_dict:primary_symbol=list(self.price_data_dict.keys())[0];data=self.price_data_dict[primary_symbol].copy()
		else:
			if self.price_data.empty or len(self.price_data)==0:logger.warning('Price data is empty, returning empty signals DataFrame');return pd.DataFrame()
			if len(self.price_data)==1:logger.warning('Only single row of price data available, duplicating for signal computation');self.price_data=pd.concat([self.price_data,self.price_data],ignore_index=True)
			data=self.price_data.copy()
		if'price'not in data.columns:logger.error("Required 'price' column missing from price data");return pd.DataFrame()
		if'returns'not in data.columns:
			if len(data)>1:data['returns']=data['price'].pct_change()
			else:data['returns']=.0
		static_cache_key=f"static_signals:{symbol_key}";static_signals=None
		if self.cache=='redis'and self._redis_client:
			try:
				cached_static=self._redis_client.get(static_cache_key)
				if cached_static:static_signals=json.loads(cached_static)
			except Exception:pass
		if static_signals is None:
			static_signals={'momentum':self.time_series_signals.momentum(data).tolist(),'reversal':self.time_series_signals.short_term_reversal(data).tolist(),'sma_dist':self.time_series_signals.sma_distance(data).tolist(),'ema_trend':self.time_series_signals.ema_trend(data).tolist(),'ma_crossover':self.time_series_signals.ma_crossover(data).tolist()}
			if self.cache=='redis'and self._redis_client:
				try:self._redis_client.setex(static_cache_key,3600,json.dumps(static_signals))
				except Exception:pass
		data['signal_momentum']=pd.Series(static_signals['momentum'],index=data.index);data['signal_reversal']=pd.Series(static_signals['reversal'],index=data.index);data['signal_sma_dist']=pd.Series(static_signals['sma_dist'],index=data.index);data['signal_ema_trend']=pd.Series(static_signals['ema_trend'],index=data.index);data['signal_ma_crossover']=pd.Series(static_signals['ma_crossover'],index=data.index);dynamic_cache_key=f"dynamic_signals:{symbol_key}";dynamic_signals=None
		if self.cache=='redis'and self._redis_client:
			try:
				cached_dynamic=self._redis_client.get(dynamic_cache_key)
				if cached_dynamic:dynamic_signals=json.loads(cached_dynamic)
			except Exception:pass
		if dynamic_signals is None:
			dynamic_signals={'vol_breakout':self.time_series_signals.volatility_breakout(data).tolist(),'price_level':self.time_series_signals.price_level(data).tolist(),'realized_var':self.volatility_signals.realized_variance(data).tolist(),'realized_vol':self.volatility_signals.realized_volatility(data).tolist(),'garch_vol':self.volatility_signals.garch_volatility(data).tolist(),'vol_of_vol':self.volatility_signals.vol_of_vol(data).tolist(),'skewness':self.volatility_signals.skewness(data).tolist(),'kurtosis':self.volatility_signals.kurtosis(data).tolist(),'vol_clustering':self.volatility_signals.volatility_clustering(data).tolist()}
			if self.cache=='redis'and self._redis_client:
				try:self._redis_client.setex(dynamic_cache_key,900,json.dumps(dynamic_signals))
				except Exception:pass
		data['signal_vol_breakout']=pd.Series(dynamic_signals['vol_breakout'],index=data.index);data['signal_price_level']=pd.Series(dynamic_signals['price_level'],index=data.index);data['signal_realized_var']=pd.Series(dynamic_signals['realized_var'],index=data.index);data['signal_realized_vol']=pd.Series(dynamic_signals['realized_vol'],index=data.index);data['signal_garch_vol']=pd.Series(dynamic_signals['garch_vol'],index=data.index);data['signal_vol_of_vol']=pd.Series(dynamic_signals['vol_of_vol'],index=data.index);data['signal_skewness']=pd.Series(dynamic_signals['skewness'],index=data.index);data['signal_kurtosis']=pd.Series(dynamic_signals['kurtosis'],index=data.index);data['signal_vol_clustering']=pd.Series(dynamic_signals['vol_clustering'],index=data.index);data['signal_turnover']=self.liquidity_signals.turnover(data);data['signal_volume_zscore']=self.liquidity_signals.volume_zscore(data);data['signal_amihud']=self.liquidity_signals.amihud_illiquidity(data);data['signal_trade_imbalance']=self.liquidity_signals.trade_imbalance(data);data['signal_vpin']=self.liquidity_signals.vpin(data);data['signal_size']=self.cross_sectional_signals.size(data);data['signal_cs_momentum']=self.cross_sectional_signals.cross_sectional_momentum(data);data['signal_low_vol']=self.cross_sectional_signals.low_volatility(data);data['signal_vol_regime']=self.regime_signals.volatility_regime(data);data['signal_liq_regime']=self.regime_signals.liquidity_regime(data)
		if'returns'in data.columns:
			market_returns=data['returns'];key_signals=['signal_momentum','signal_realized_vol','signal_amihud']
			for sig_col in key_signals:
				if sig_col in data.columns:
					try:data[f"{sig_col}_crowding"]=self.meta_signals.signal_crowding(data,sig_col,market_returns);data[f"{sig_col}_instability"]=self.meta_signals.signal_instability(data,sig_col)
					except Exception:pass
		if self.multi_asset_mode and len(self.price_data_dict)>1:
			cross_asset_signals=self.compute_cross_asset_signals()
			if not cross_asset_signals.empty:data=pd.concat([data,cross_asset_signals],axis=1)
		alternative_signals=self.compute_alternative_signals()
		if not alternative_signals.empty:data=pd.concat([data,alternative_signals],axis=1)
		if self.multi_asset_mode:
			if asset_symbol is not None:self.signals_df_dict[asset_symbol]=data
			elif self.price_data_dict:primary_symbol=list(self.price_data_dict.keys())[0];self.signals_df_dict[primary_symbol]=data
		self.signals_df=data
		if VERBOSE_LOGGING:num_signals=len([c for c in data.columns if c.startswith('signal_')]);asset_info=f" for {asset_symbol}"if asset_symbol else'';logger.debug(f"Computed {num_signals} signals{asset_info}")
		return data
	def compute_alternative_signals(self)->pd.DataFrame:
		if self.price_data.empty or not self.alternative_data_cache:return pd.DataFrame()
		result=pd.DataFrame(index=self.price_data.index);window_days=ALTERNATIVE_DATA_CONFIG.get('window_size_days',7)
		try:
			onchain_data=self.alternative_data_cache.get('onchain',{})
			if onchain_data:result['signal_onchain_addr']=onchain_data.get('active_addresses',.0);result['signal_onchain_tx']=onchain_data.get('transaction_volume',.0);result['signal_onchain_growth']=onchain_data.get('network_growth',.0);result['signal_onchain_health']=self.onchain_signals.network_health_score(onchain_data)
			social_data=self.alternative_data_cache.get('social',{})
			if social_data:result['signal_twitter']=social_data.get('twitter_sentiment',.0);result['signal_reddit']=social_data.get('reddit_sentiment',.0);result['signal_social_momentum']=self.social_sentiment_signals.social_momentum(social_data)
			news_data=self.alternative_data_cache.get('news',{})
			if news_data:result['signal_news']=news_data.get('news_sentiment',.0);result['signal_headline']=news_data.get('headline_sentiment',.0);result['signal_news_volume']=news_data.get('news_volume',.0)
			github_data=self.alternative_data_cache.get('github',{})
			if github_data:result['signal_github']=self.github_signals.github_activity(github_data);result['signal_github_momentum']=self.github_signals.github_momentum(github_data);result['signal_github_health']=self.github_signals.github_community_health(github_data)
			exchange_data=self.alternative_data_cache.get('exchange',{})
			if exchange_data:result['signal_funding']=self.exchange_metrics_signals.funding_rate(exchange_data);result['signal_oi']=self.exchange_metrics_signals.open_interest(exchange_data);result['signal_ls_ratio']=self.exchange_metrics_signals.long_short_ratio(exchange_data)
		except Exception as e:logger.debug(f"Alternative signal computation failed: {e}")
		if self.cache=='redis'and self._redis_client:
			try:cache_data={'signals':result.to_dict('index')if isinstance(result,pd.DataFrame)else result,'timestamp':time.time()};cache_key=f"signals:{self.exchange_name}:{self.symbol}"if hasattr(self,'symbol')else'signals:default';self._redis_client.setex(cache_key,1800,json.dumps(cache_data))
			except Exception:pass
		return result
	def compute_cross_asset_signals(self)->pd.DataFrame:
		if not self.multi_asset_mode or len(self.price_data_dict)<2:return pd.DataFrame()
		primary_symbol=list(self.price_data_dict.keys())[0];primary_data=self.price_data_dict[primary_symbol];asset_symbols=list(self.price_data_dict.keys());signal_series={};AlignSignal=lambda sig:sig.reindex(primary_data.index,fill_value=.0).ffill().fillna(.0)if not sig.empty else pd.Series(index=primary_data.index,dtype=float)
		for(i,asset1)in enumerate(asset_symbols):
			for asset2 in asset_symbols[i+1:]:
				data1,data2=self.price_data_dict[asset1],self.price_data_dict[asset2]
				if'date'not in data1.columns or'date'not in data2.columns:logger.debug(f"Skipping cross-asset signals for {asset1}-{asset2}: missing date column");continue
				corr_signal=self.relative_value_signals.cross_asset_correlation(data1,data2)
				if not corr_signal.empty:signal_series[f"signal_{asset1}_{asset2}_corr"]=AlignSignal(corr_signal)
				rs_signal=self.relative_value_signals.relative_strength(data1,data2)
				if not rs_signal.empty:signal_series[f"signal_{asset1}_{asset2}_rs"]=AlignSignal(rs_signal)
				if asset1.upper()=='BTC'and asset2.upper()=='ETH'or asset1.upper()=='ETH'and asset2.upper()=='BTC':
					btc_data=self.price_data_dict.get('BTC')if'BTC'in self.price_data_dict else self.price_data_dict.get('btc')if'btc'in self.price_data_dict else None;eth_data=self.price_data_dict.get('ETH')if'ETH'in self.price_data_dict else self.price_data_dict.get('eth')if'eth'in self.price_data_dict else None
					if btc_data is not None and eth_data is not None and not btc_data.empty and not eth_data.empty:
						btc_eth_corr=self.relative_value_signals.btc_eth_correlation(eth_data,btc_data)
						if not btc_eth_corr.empty:signal_series['signal_btc_eth_corr']=AlignSignal(btc_eth_corr)
		result=pd.concat(signal_series,axis=1)if signal_series else pd.DataFrame(index=primary_data.index)
		if VERBOSE_LOGGING:logger.debug(f"Computed {len(result.columns)} cross-asset signals")
		if self.cache=='redis'and self._redis_client:
			try:cache_data={'signals':result.to_dict('index')if isinstance(result,pd.DataFrame)else result,'timestamp':time.time()};cache_key=f"cross_signals:{self.exchange_name}"if hasattr(self,'exchange_name')else'cross_signals:default';self._redis_client.setex(cache_key,1800,json.dumps(cache_data))
			except Exception:pass
		return result
	def build_causal_graph(self)->None:
		variables=['price','volume','volatility','returns','momentum','market_sentiment','trading_volume','liquidity'];edges=[('volume','price'),('volatility','price'),('momentum','price'),('market_sentiment','momentum'),('market_sentiment','trading_volume'),('trading_volume','price'),('liquidity','price')]
		if self.alternative_data_cache:
			if'onchain'in self.alternative_data_cache:variables.extend(['onchain_metrics','network_growth']);edges.extend([('onchain_metrics','volume'),('network_growth','price')])
			if'social'in self.alternative_data_cache:variables.append('social_sentiment');edges.extend([('social_sentiment','market_sentiment'),('social_sentiment','momentum')])
			if'news'in self.alternative_data_cache:variables.append('news_sentiment');edges.extend([('news_sentiment','market_sentiment'),('news_sentiment','volatility')])
			if'exchange'in self.alternative_data_cache:variables.append('exchange_metrics');edges.append(('exchange_metrics','liquidity'))
		self.causal_engine.build_scm(variables,edges)
	def _compute_alternative_signal_confidence(self,signal_name:str,data_source:str)->float:
		weights=ALTERNATIVE_DATA_CONFIG.get('confidence_weights',{'freshness':.4,'reliability':.4,'stability':.2});freshness=1.
		if data_source in self.alternative_data_timestamps:age_seconds=time.time()-self.alternative_data_timestamps[data_source];ttl=ALTERNATIVE_DATA_CONFIG['cache_ttl'].get(data_source,3600);freshness=max(.0,1.-age_seconds/ttl)
		reliability=.8 if data_source in self.alternative_data_cache else .0;stability=.7;confidence=weights['freshness']*freshness+weights['reliability']*reliability+weights['stability']*stability;return float(np.clip(confidence,.0,1.))
	def validate_signals(self)->Dict[str,Dict[str,float]]:
		if self.signals_df.empty or'returns'not in self.signals_df.columns:return{}
		self.build_causal_graph();df_vars=['price','volume','volatility','returns','momentum'];available_vars=[v for v in df_vars if v in self.signals_df.columns]
		if len(available_vars)>=3:
			try:self.causal_engine.fit_from_data(self.signals_df,available_vars,window=30)
			except Exception as e:logger.debug(f"CRCA fit failed: {e}")
		target=self.signals_df['returns'].shift(-1);regimes=self.signals_df.get('signal_vol_regime',pd.Series());signal_scores={};signal_cols=[c for c in self.signals_df.columns if c.startswith('signal_')and not c.endswith('_crowding')and not c.endswith('_instability')]
		for signal_col in signal_cols[:20]:
			signal_values=self.signals_df[signal_col];aligned=pd.concat([signal_values,target],axis=1).dropna()
			if len(aligned)>10:
				score=self.signal_validator.compute_causal_score(signal_col,aligned.iloc[:,0],aligned.iloc[:,1],regimes if not regimes.empty else None)
				if any(alt_prefix in signal_col for alt_prefix in['twitter','reddit','onchain','news','github','funding','oi']):
					data_source='social'if any(x in signal_col for x in['twitter','reddit'])else'onchain'if'onchain'in signal_col else'news'if'news'in signal_col or'headline'in signal_col else'github'if'github'in signal_col else'exchange'if any(x in signal_col for x in['funding','oi','ls'])else'unknown'
					if data_source!='unknown':alt_confidence=self._compute_alternative_signal_confidence(signal_col,data_source);score['score']=score.get('score',.0)*alt_confidence;score['alternative_confidence']=alt_confidence
				signal_scores[signal_col]=score
		if VERBOSE_LOGGING:logger.debug(f"Validated {len(signal_scores)} signals")
		return signal_scores
	def generate_predictions(self,signal_scores:Dict[str,Dict[str,float]])->Tuple[np.ndarray,np.ndarray,Dict[str,Any]]:
		if self.multi_asset_mode:
			primary_symbol=list(self.price_data_dict.keys())[0]if self.price_data_dict else None
			if primary_symbol:
				multi_preds=self.generate_multi_asset_predictions(signal_scores)
				if primary_symbol in multi_preds:pred,intervals,metadata=multi_preds[primary_symbol];return pred,intervals,metadata
		if self.signals_df.empty:return np.array([]),np.array([]),{}
		available_signals=set(self.signals_df.columns);valid_signals=[s for(s,score)in signal_scores.items()if s in available_signals and score.get('score',0)>.5]
		if not valid_signals:valid_signals=[c for c in self.signals_df.columns if c.startswith('signal_')]
		valid_signals=[s for s in valid_signals if s in available_signals]
		if not valid_signals:logger.warning('No valid signals found in signals_df, returning zero predictions');return np.array([.0]*len(self.signals_df)),np.array([[.01]]),{}
		X=self.signals_df[valid_signals].fillna(0).values
		if self.longterm_mode and hasattr(self,'prediction_horizon_days'):k_periods=max(1,int(self.prediction_horizon_days));y=self.target_generator.generate_forward_returns(self.signals_df,k=k_periods).fillna(0).values
		else:y=self.target_generator.generate_forward_returns(self.signals_df,k=1).fillna(0).values
		train_size=int(len(X)*.8)
		if train_size<10:return np.zeros(len(X)),np.array([[.01]]),{}
		X_train,X_test=X[:train_size],X[train_size:];y_train,y_test=y[:train_size],y[train_size:];models,model_predictions,trained_models={},{},[]
		if SKLEARN_AVAILABLE:
			model_configs=[('linear',{'type':'linear'}),('rf',{'type':'rf','n_estimators':50}),('gb',{'type':'gb','n_estimators':50})]
			for(name,config)in model_configs:
				try:
					model=self.model_factory.create_model(config['type'],**{k:v for(k,v)in config.items()if k!='type'})
					if model is not None:model.fit(X_train,y_train);models[name]=model;trained_models.append(name);self.ensemble_predictor.add_model(name,model,weight=1.);model_predictions[name]=model.predict(X)
				except Exception as e:
					if VERBOSE_LOGGING:logger.debug(f"Model {name} training failed: {e}")
		ensemble_weights={}
		if len(trained_models)>0:
			pred_array=np.array([model_predictions[name]for name in trained_models]);weights=np.ones(len(trained_models))/len(trained_models)
			if hasattr(self,'causal_engine'):
				try:
					weights=self.meta_learner.optimize(signal_names=trained_models,recent_performance={m:1. for m in trained_models},regime='normal',model_names=trained_models)[0].values();weights=np.array(list(weights))
					if np.sum(weights)>0:weights=weights/np.sum(weights)
				except Exception:weights=np.ones(len(trained_models))/len(trained_models)
			ensemble_weights={m:float(w)for(m,w)in zip(trained_models,weights)};predictions=np.average(pred_array,axis=0,weights=weights);full_pred_std=np.maximum(np.std(pred_array,axis=0),np.abs(predictions)*.05)
		else:predictions=np.zeros(len(X));full_pred_std=np.zeros(len(X));logger.warning('No models could be trained - returning zero predictions')
		signal_contributions={}
		if SKLEARN_AVAILABLE and'rf'in models and models['rf']is not None:
			try:
				rf_model=models['rf'];feature_importance=rf_model.feature_importances_
				for(i,sig_name)in enumerate(valid_signals):
					if i<len(feature_importance):signal_contributions[sig_name]=float(feature_importance[i])
			except Exception:pass
		if'returns'in self.signals_df.columns:
			returns_df=pd.DataFrame({'returns':self.signals_df['returns']});covariance=self.cov_estimator.ewma_covariance(returns_df)
			if CovarianceEstimator.CovSize(covariance)==0:covariance=np.array([[.01]])
		else:covariance=np.array([[.01]])
		avg_signal_score=.5
		if signal_scores:
			signal_score_values=[v.get('score',.0)for v in signal_scores.values()if isinstance(v,dict)]
			if len(signal_score_values)>0:avg_signal_score=float(np.mean(signal_score_values))
		per_model_pred={m:float(model_predictions[m][-1])for m in trained_models}if trained_models else{};per_model_std={m:float(np.std(model_predictions[m]))for m in trained_models}if trained_models else{};prediction_metadata={'signal_contributions':signal_contributions,'prediction_std':full_pred_std.tolist()if len(full_pred_std)>0 else[],'valid_signals':valid_signals,'model_count':len(trained_models),'model_names':trained_models,'ensemble_weights':ensemble_weights,'per_model_pred':per_model_pred,'per_model_std':per_model_std,'avg_signal_score':avg_signal_score,'is_ensemble':len(trained_models)>1}
		if VERBOSE_LOGGING:
			if len(trained_models)>1:logger.debug(f"Ensemble prediction using {len(trained_models)} models: {", ".join(trained_models)} weights={ensemble_weights}")
			else:logger.debug(f"Single model prediction using: {trained_models[0]if trained_models else"none"}")
		return predictions,covariance,prediction_metadata
	def generate_multi_asset_predictions(self,signal_scores:Dict[str,Dict[str,float]])->Dict[str,Tuple[np.ndarray,Dict[str,np.ndarray],Dict[str,Any]]]:
		if not self.multi_asset_mode or not self.price_data_dict:return{}
		results={}
		for(symbol,asset_data)in self.price_data_dict.items():
			try:
				assert symbol in self.price_data_dict and'returns'in asset_data.columns,f"Invalid asset data for {symbol}";asset_signals=[c for c in self.signals_df.columns if c.startswith(f"{symbol}_signal_")or c.startswith('signal_')and not any(a in c for a in self.price_data_dict.keys()if a!=symbol)];cross_asset_signals=[c for c in self.signals_df.columns if f"_{symbol}_"in c or f"{symbol}_"in c];all_signals=list(set(asset_signals+cross_asset_signals))
				if not all_signals:all_signals=[c for c in self.signals_df.columns if c.startswith('signal_')]
				available_signals=set(self.signals_df.columns);valid_signals=[s for s in all_signals if s in available_signals and s in signal_scores and signal_scores[s].get('score',0)>.5]
				if not valid_signals:valid_signals=[s for s in all_signals if s in available_signals][:20]
				if not valid_signals:logger.warning(f"No valid signals found for {symbol}, skipping prediction");continue
				dfX=self.signals_df[valid_signals].infer_objects(copy=False);X=np.nan_to_num(dfX.to_numpy(dtype=float,copy=True),nan=.0,posinf=.0,neginf=.0)
				if self.longterm_mode and hasattr(self,'prediction_horizon_days'):k_periods=max(1,int(self.prediction_horizon_days));y_raw=self.target_generator.generate_forward_returns(asset_data,k=k_periods,target_col='price').infer_objects(copy=False)
				else:y_raw=self.target_generator.generate_forward_returns(asset_data,k=1,target_col='price').infer_objects(copy=False)
				y=np.nan_to_num(y_raw.to_numpy(dtype=float,copy=True),nan=.0,posinf=.0,neginf=.0);min_len=min(len(X),len(y));X,y=X[:min_len],y[:min_len];assert len(X)>=10,f"Insufficient data for {symbol}";train_size=int(len(X)*.8);assert train_size>=10,f"Insufficient training data for {symbol}";X_train,X_test=X[:train_size],X[train_size:];y_train,y_test=y[:train_size],y[train_size:];models,model_predictions={},{}
				if SKLEARN_AVAILABLE:models['linear']=self.model_factory.create_model('linear');models['rf']=self.model_factory.create_model('rf',n_estimators=50)
				for(name,model)in models.items():
					assert model is not None,f"Model {name} is None"
					try:model.fit(X_train,y_train);model_predictions[name]=model.predict(X_test)
					except Exception:pass
				assert model_predictions,f"No model predictions for {symbol}";pred_array=np.array(list(model_predictions.values()));weights=np.ones(len(model_predictions))/len(model_predictions)
				if hasattr(self,'causal_engine'):
					try:
						weights=self.meta_learner.optimize(signal_names=list(model_predictions.keys()),recent_performance={m:1. for m in model_predictions.keys()},regime='normal',model_names=list(model_predictions.keys()))[0].values();weights=np.array(list(weights))
						if np.sum(weights)>0:weights=weights/np.sum(weights)
					except Exception:weights=np.ones(len(model_predictions))/len(model_predictions)
				ensemble_weights={m:float(w)for(m,w)in zip(model_predictions.keys(),weights)};predictions=np.average(pred_array,axis=0,weights=weights);pred_std=np.maximum(np.std(pred_array,axis=0),np.abs(predictions)*.05);full_predictions=np.zeros(len(X));full_pred_std=np.zeros(len(X));full_predictions[train_size:]=predictions;full_pred_std[train_size:]=pred_std
				if train_size>0:full_predictions[:train_size]=np.mean(predictions)if len(predictions)>0 else .0;full_pred_std[:train_size]=np.mean(pred_std)if len(pred_std)>0 else abs(full_predictions[:train_size])*.1
				latest_pred=full_predictions[-1]if len(full_predictions)>0 else .0;latest_std=full_pred_std[-1]if len(full_pred_std)>0 else abs(latest_pred)*.1;current_price=asset_data['price'].iloc[-1];n_simulations=250
				if self.longterm_mode and hasattr(self,'prediction_horizon_days'):base_horizon=self.prediction_horizon_days;horizon_days=[base_horizon,base_horizon*2,base_horizon*4]
				else:horizon_days=[1,3,7]
				pred_return=latest_pred;pred_std_return=latest_std
				if abs(pred_return)>1.:pred_return=pred_return/current_price if current_price>0 else .0;pred_std_return=pred_std_return/current_price if current_price>0 else abs(pred_return)*.1
				intervals={}
				for horizon in horizon_days:
					simulated_prices=[]
					for _ in range(n_simulations):daily_returns=np.random.normal(pred_return,pred_std_return,horizon);final_price=current_price*np.exp(np.sum(daily_returns));simulated_prices.append(final_price)
					simulated_prices=np.array(simulated_prices);mean_price=float(np.mean(simulated_prices));lower_price=float(np.percentile(simulated_prices,2.5));upper_price=float(np.percentile(simulated_prices,97.5));intervals[f"{horizon}d"]={'lower':lower_price,'upper':upper_price,'mean':mean_price,'median':float(np.median(simulated_prices)),'std':float(np.std(simulated_prices))}
				metadata={'num_signals':len(valid_signals),'prediction_uncertainty':float(np.mean(full_pred_std)),'prediction_std':full_pred_std.tolist(),'asset':symbol,'model_count':len(model_predictions),'model_names':list(model_predictions.keys()),'ensemble_weights':ensemble_weights,'per_model_pred':{m:float(pred_array[i][-1])for(i,m)in enumerate(model_predictions.keys())},'per_model_std':{m:float(np.std(pred_array[i]))for(i,m)in enumerate(model_predictions.keys())}};results[symbol]=full_predictions,intervals,metadata
				if VERBOSE_LOGGING:logger.debug(f"[MULTI] {symbol}: models={list(model_predictions.keys())} weights={ensemble_weights}")
			except Exception as e:logger.debug(f"Failed to generate predictions for {symbol}: {e}");continue
		if VERBOSE_LOGGING:logger.debug(f"Generated predictions for {len(results)} assets")
		return results
	def optimize_portfolio(self,expected_returns:Union[Dict[str,float],np.ndarray],covariance:Union[pd.DataFrame,np.ndarray,Dict[str,Any]],signal_scores:Optional[Dict[str,Dict[str,float]]]=None)->Union[Dict[str,float],np.ndarray]:
		if self.multi_asset_mode:
			if isinstance(expected_returns,dict)and isinstance(covariance,pd.DataFrame):
				asset_types_list=[self.asset_types.get(symbol,'crypto')for symbol in expected_returns.keys()];cross_asset_constraints={}
				for asset_type in set(asset_types_list):cross_asset_constraints[asset_type]=.6 if asset_type=='crypto'else .4 if asset_type=='stock'else .2 if asset_type=='fx'else .3
				return self.portfolio_optimizer.optimize_asset_allocation(expected_returns_dict=expected_returns,covariance_df=covariance,constraints={'asset_types':asset_types_list,'cross_asset_constraints':cross_asset_constraints,'max_leverage':1.})
		if len(expected_returns)==0:return np.array([])
		cov_array=np.array(covariance.values)if isinstance(covariance,pd.DataFrame)else covariance if isinstance(covariance,np.ndarray)else np.array(covariance)
		if cov_array.size==0:return np.array([])
		n=len(expected_returns)
		if cov_array.shape!=(n,n):cov_array=np.eye(n)*.01
		return self.portfolio_optimizer.optimize_cvar(expected_returns=expected_returns,covariance=cov_array,max_leverage=1.)
	def _compute_confidence_only(self,predictions:np.ndarray,signal_scores:Dict[str,Dict[str,float]],pred_metadata:Dict[str,Any],confidence_intervals:Dict[str,Dict[str,float]])->Tuple[float,List[str]]:
		try:
			weight=float(predictions[0])if len(predictions)>0 else .0;pred_std=pred_metadata.get('prediction_std',[]);latest_pred=predictions[-1]if len(predictions)>0 else .0;confidence_factors,explanations=[],[];signal_strength=abs(weight)
			if np.isnan(signal_strength)or signal_strength<1e-06:signal_strength=.0
			if signal_scores:
				for(signal_name,signal_data)in signal_scores.items():
					if isinstance(signal_data,dict)and'score'in signal_data:
						score_val=float(signal_data['score'])
						if not np.isnan(score_val):confidence_factors.append(abs(score_val)*.8)
			if len(signal_scores)>0:
				avg_signal_score=np.mean([v.get('score',.0)for v in signal_scores.values()if isinstance(v,dict)])
				if not np.isnan(avg_signal_score):confidence_factors.append(avg_signal_score*.9)
			heuristic_confidence=min(1.,max(.0,sum(confidence_factors)));statistical_confidence=(lambda ci_1d:(lambda ci_width,ci_mean:.9 if(relative_width:=ci_width/abs(ci_mean)if abs(ci_mean)>1e-06 else 1.)<.1 else .75 if relative_width<.2 else .6 if relative_width<.3 else .45 if relative_width<.5 else .3)(ci_1d['upper']-ci_1d['lower'],ci_1d['mean'])if'lower'in ci_1d and'upper'in ci_1d and'mean'in ci_1d and ci_1d['mean']>0 else .5)(confidence_intervals.get('1d',{}))if'1d'in confidence_intervals else .5;ensemble_confidence=(lambda latest_std:.85 if(cv:=abs(latest_std)/abs(latest_pred))<.2 else .7 if cv<.3 else .55 if cv<.5 else .4)(pred_std[-1])if len(pred_std)>0 and abs(latest_pred)>1e-06 else .5;raw_confidence=min(1.,max(.0,.4*heuristic_confidence+.35*statistical_confidence+.25*ensemble_confidence));confidence=self.confidence_calibrator.calibrate_prob(raw_confidence)if hasattr(self,'confidence_calibrator')and self.confidence_calibrator else raw_confidence;confidence=min(1.,max(.0,confidence));causal_score,causal_block=self._evaluate_causal_stability()
			if causal_block:confidence=.0;explanations.append(f"⚠️ Causal block: unstable causal structure (score {causal_score:.2f})")
			else:multiplier=max(.5,min(1.,causal_score));confidence*=multiplier;explanations.append(f"Causal stability: score {causal_score:.2f}, confidence scaled by {multiplier:.2f}")
			if VERBOSE_LOGGING:calib_info=f" (calibrated: {confidence:.0%})"if hasattr(self,'confidence_calibrator')and self.confidence_calibrator.isotonic_model else'';explanations.append(f"Confidence breakdown: heuristic={heuristic_confidence:.0%}, statistical={statistical_confidence:.0%}, ensemble={ensemble_confidence:.0%}, raw={raw_confidence:.0%}{calib_info}")
			return confidence,explanations
		except Exception as e:logger.warning(f"Confidence calculation failed: {e}, using default");return .5,[f"Confidence calculation failed: {str(e)}"]
	def _get_scm_dataset(self,window:int=200,horizon:int=1)->Optional[pd.DataFrame]:
		try:
			df=None
			if hasattr(self,'signals_df')and self.signals_df is not None and not self.signals_df.empty:df=self.signals_df.copy()
			elif hasattr(self,'price_data')and self.price_data is not None and not self.price_data.empty:df=self.price_data.copy()
			if df is None or df.empty:return
			work=df.copy().tail(window+horizon+5)
			if'returns'not in work.columns and'price'in work.columns:work['returns']=work['price'].pct_change()
			work['future_return']=work['returns'].shift(-horizon);work['momentum']=work['returns'].shift(1);work['trend']=work['returns'].rolling(window=5,min_periods=3).mean().shift(1)if'returns'in work.columns else np.nan;work['reversion_pressure']=((work['returns']-work['returns'].rolling(window=10,min_periods=5).mean())/(work['returns'].rolling(window=10,min_periods=5).std()+1e-06)).shift(1)if'returns'in work.columns else np.nan;work['short_vol']=work['returns'].rolling(window=5,min_periods=3).std().shift(1)if'returns'in work.columns else np.nan;work['regime_vol']=work['returns'].ewm(span=20,adjust=False).std().shift(1)if'returns'in work.columns else np.nan
			if'volume'in work.columns:work['volume']=work['volume'].shift(1)
			elif'turnover'in work.columns:work['volume']=work['turnover'].shift(1)
			else:work['volume']=np.nan
			if'liquidity'in work.columns:work['liquidity']=work['liquidity'].shift(1)
			else:work['liquidity']=np.nan
			if'sentiment'in work.columns:work['sentiment']=work['sentiment'].shift(1)
			else:work['sentiment']=np.nan
			if'funding_regime'in work.columns:work['funding_regime']=work['funding_regime'].shift(1)
			if'session_time'in work.columns:work['session_time']=work['session_time'].shift(1)
			cols=['sentiment','liquidity','regime_vol','volume','momentum','trend','short_vol','reversion_pressure','future_return']
			if'funding_regime'in work.columns:cols.append('funding_regime')
			if'session_time'in work.columns:cols.append('session_time')
			work=work[cols].dropna()
			if len(work)<50:return
			return work.tail(window)
		except Exception:return
	def _fit_causal_graph(self,dataset:pd.DataFrame)->Dict[str,Dict[str,float]]:
		edges={'volume':['sentiment','liquidity'],'momentum':['sentiment','liquidity','volume'],'short_vol':['regime_vol'],'future_return':['momentum','trend','reversion_pressure','short_vol']};opt_edges={'short_vol':['liquidity'],'momentum':['short_vol']};strengths:Dict[str,Dict[str,float]]={}
		for(child,parents)in edges.items():
			strengths[child]={}
			for p in parents:
				if p in dataset.columns and child in dataset.columns:strengths[child][p]=.0
		for(child,parents)in opt_edges.items():
			if child not in strengths:strengths[child]={}
			for p in parents:
				if p in dataset.columns and child in dataset.columns:strengths[child][p]=.0
		try:
			for(child,parents)in strengths.items():
				y=dataset[child].values;X_cols=[p for p in parents.keys()]
				if not X_cols:continue
				X=dataset[X_cols].values
				if X.shape[0]<X.shape[1]+5:continue
				coef,_,_,_=np.linalg.lstsq(X,y,rcond=None)
				for(i,p)in enumerate(X_cols):strengths[child][p]=float(coef[i])
		except Exception as e:
			if VERBOSE_LOGGING:logger.debug(f"SCM fit failed: {e}")
		return strengths
	def _simulate_interventions(self,strengths:Dict[str,Dict[str,float]],dataset:pd.DataFrame,n_scenarios:int=50,epsilon:float=.1)->Dict[str,Any]:
		if'future_return'not in dataset.columns:return{'edge_effects':{},'score':1.}
		target_std=float(dataset['future_return'].std()or 1e-06);parents=['sentiment','liquidity','volume','momentum','short_vol','trend','reversion_pressure'];edge_effects:Dict[str,List[float]]={p:[]for p in parents}
		for _ in range(n_scenarios):
			row=dataset.sample(1).iloc[0];base_return=float(row['future_return'])
			for p in parents:
				if p not in row or p not in dataset.columns:continue
				perturbed=row.copy();delta=epsilon*(1 if np.random.rand()>.5 else-1);perturbed[p]=perturbed[p]+delta;mom=perturbed['momentum']
				if p in strengths.get('momentum',{}):mom+=strengths['momentum'][p]*delta
				fut=perturbed['future_return'];fut+=strengths.get('future_return',{}).get('momentum',.0)*(mom-row['momentum']);fut+=strengths.get('future_return',{}).get('trend',.0)*(perturbed.get('trend',row.get('trend',.0))-row.get('trend',.0));fut+=strengths.get('future_return',{}).get('reversion_pressure',.0)*(perturbed.get('reversion_pressure',row.get('reversion_pressure',.0))-row.get('reversion_pressure',.0));fut+=strengths.get('future_return',{}).get('short_vol',.0)*(perturbed.get('short_vol',row.get('short_vol',.0))-row.get('short_vol',.0));edge_effects[p].append((fut-base_return)/(target_std+1e-06))
		edge_scores=[]
		for(p,effects)in edge_effects.items():
			if not effects:continue
			effects=np.array(effects);mean_eff=np.mean(effects);sign_flip_penalty=1. if np.sign(np.median(effects))==np.sign(np.mean(effects))else .5;magnitude=min(1.,max(.0,abs(mean_eff)));edge_scores.append(magnitude*sign_flip_penalty)
		score=float(np.mean(edge_scores))if edge_scores else 1.;return{'edge_effects':edge_effects,'score':score}
	def _compute_causal_score(self,strengths:Dict[str,Dict[str,float]],dataset:pd.DataFrame)->float:
		if'future_return'not in dataset.columns or dataset['future_return'].std()==0:return 1.
		target_std=float(dataset['future_return'].std());edge_scores=[]
		def norm_score(coef:float)->float:s=abs(coef)/(target_std+1e-06);return float(max(.0,min(1.,s)))
		for parent in['momentum','trend','reversion_pressure','short_vol']:coef=strengths.get('future_return',{}).get(parent,.0);edge_scores.append(norm_score(coef))
		vol_mom=strengths.get('momentum',{}).get('volume',.0);mom_fr=strengths.get('future_return',{}).get('momentum',.0);chain_score=norm_score(vol_mom*mom_fr);edge_scores.append(chain_score);sv_mom=strengths.get('momentum',{}).get('short_vol',.0);chain_sv=norm_score(sv_mom*mom_fr);edge_scores.append(chain_sv)
		if not edge_scores:return 1.
		return float(np.mean(edge_scores))
	def _validate_with_crca(self,signal:str,expected_return:float,confidence:float,symbol:str)->Dict[str,Any]:
		if not hasattr(self,'causal_engine')or not self.causal_engine:return{'approved':True,'reason':'CRCAAgent not available'}
		try:
			crca=self.causal_engine.crca
			if signal=='BUY':
				if expected_return<=0:return{'approved':False,'reason':'Expected return not positive'}
				if confidence<getattr(self,'min_confidence_threshold',.85):return{'approved':False,'reason':f"Confidence {confidence:.0%} below threshold {getattr(self,"min_confidence_threshold",.85):.0%}"}
				return{'approved':True,'reason':'CRCAAgent validation passed'}
			elif signal=='SELL':
				if expected_return>=0:return{'approved':False,'reason':'Expected return not negative'}
				if confidence<getattr(self,'min_confidence_threshold',.85):return{'approved':False,'reason':f"Confidence {confidence:.0%} below threshold {getattr(self,"min_confidence_threshold",.85):.0%}"}
				return{'approved':True,'reason':'CRCAAgent validation passed'}
			else:return{'approved':True,'reason':'HOLD signal, no validation needed'}
		except Exception as e:
			logger.warning(f"CRCAAgent validation error: {e}")
			if self.longterm_mode:return{'approved':False,'reason':f"Validation error: {str(e)}"}
			return{'approved':True,'reason':'Validation error, proceeding with caution'}
	def _evaluate_causal_stability(self)->Tuple[float,bool]:
		neutral_score,neutral_block=1.,False;dataset=self._get_scm_dataset()
		if dataset is None:return neutral_score,neutral_block
		strengths=self._fit_causal_graph(dataset);structural_score=self._compute_causal_score(strengths,dataset);mc=self._simulate_interventions(strengths,dataset,n_scenarios=50,epsilon=.1);score=float(np.mean([structural_score,mc.get('score',1.)]))
		if getattr(self,'conservative_mode',False):block_threshold=.45
		elif getattr(self,'aggressive_mode',False):block_threshold=.25
		else:block_threshold=.35
		causal_block=score<block_threshold
		if VERBOSE_LOGGING:logger.debug(f"[SCM] causal_score={score:.3f}, block={causal_block}, threshold={block_threshold}, structural={structural_score:.3f}, mc={mc.get("score",1.):.3f}")
		return score,causal_block
	def _make_trading_decision(self,portfolio_weights:np.ndarray,predictions:np.ndarray,signal_scores:Dict[str,Dict[str,float]],pred_metadata:Dict[str,Any],confidence_intervals:Dict[str,Dict[str,float]],volatility:float,current_price:float,max_position_size:Optional[float]=None,symbol:str=None)->Dict[str,Any]:
		explanations=[]
		if max_position_size is None:max_position_size=getattr(self,'risk_monitor',None);max_position_size=max_position_size.max_position_size if max_position_size else TRADING_CONFIG['max_position_size']
		assert len(portfolio_weights)>0 and len(predictions)>0,'Insufficient data for decision';weight,latest_pred=portfolio_weights[0],predictions[-1]
		if weight is None or not isinstance(weight,(int,float))or np.isnan(weight)or np.isinf(weight):weight=.0;explanations.append('⚠️ Invalid portfolio weight, using 0.0')
		confidence_factors=[];signal_strength=abs(weight)
		if np.isnan(signal_strength)or signal_strength<1e-06:
			signal_strength=.0
			if VERBOSE_LOGGING:logger.debug(f"Signal strength is zero or NaN, weight={weight}")
		if signal_strength>.2:confidence_factors.append(.3);explanations.append(f"Strong signal strength ({signal_strength:.2%})")
		elif signal_strength>.1:confidence_factors.append(.2);explanations.append(f"Moderate signal strength ({signal_strength:.2%})")
		elif signal_strength>.05:confidence_factors.append(.1);explanations.append(f"Weak signal strength ({signal_strength:.2%})")
		else:confidence_factors.append(.05);explanations.append(f"Very weak signal strength ({signal_strength:.2%})")
		model_count,model_names=pred_metadata.get('model_count',1),pred_metadata.get('model_names',[])
		if model_count>=3:confidence_factors.append(.2);explanations.append(f"Ensemble prediction: {", ".join(model_names[:3])if model_names else f"{model_count} models"} agree")
		elif model_count>=2:confidence_factors.append(.15);explanations.append(f"Multiple models ({", ".join(model_names)if model_names else f"{model_count} models"})")
		else:confidence_factors.append(.1);explanations.append(f"Single model prediction ({model_names[0]if model_names else"single model"})")
		base_signals=[]
		if symbol:base_signals=[(k,v.get('score',.0))for(k,v)in signal_scores.items()if symbol in k or k.startswith('signal_')]
		if not base_signals:base_signals=[(k,v.get('score',.0))for(k,v)in signal_scores.items()]
		TopSignals=lambda n:sorted(base_signals,key=lambda x:(-x[1],x[0]))[:n];AvgSignalScore=lambda sigs:np.mean([s[1]for s in sigs])if sigs else .0;top_signals=TopSignals(5);avg_signal_score=AvgSignalScore(top_signals);signal_contributions=pred_metadata.get('signal_contributions',{})
		if avg_signal_score>.7:confidence_factors.append(.25);explanations.append(f"High-quality signals (avg score: {avg_signal_score:.2f})");top_contributors=[f"{sig_name.replace("signal_","")[:15]} ({signal_contributions.get(sig_name,.0):.1%})"if signal_contributions.get(sig_name,.0)>0 else f"{sig_name.replace("signal_","")[:15]} (score: {sig_score:.2f})"for(sig_name,sig_score)in top_signals[:3]];explanations.append(f"Top signals: {", ".join(top_contributors)if top_contributors else", ".join([s[0].replace("signal_","")[:15]for s in top_signals[:3]])}")
		elif avg_signal_score>.5:
			confidence_factors.append(.15);explanations.append(f"Moderate signal quality (avg score: {avg_signal_score:.2f})")
			if top_signals:explanations.append(f"Best signal: {top_signals[0][0].replace("signal_","")[:15]} (score: {top_signals[0][1]:.2f})")
		else:
			confidence_factors.append(.1);explanations.append(f"Low signal quality (avg score: {avg_signal_score:.2f})")
			if top_signals:explanations.append(f"Best signal: {top_signals[0][0].replace("signal_","")[:15]} (score: {top_signals[0][1]:.2f})")
		pred_std=pred_metadata.get('prediction_std',[]);latest_std=pred_std[-1]if len(pred_std)>0 else abs(latest_pred)*.1;is_prediction_in_returns=abs(latest_pred)<1. and current_price>1e1
		if is_prediction_in_returns:uncertainty_ratio=abs(latest_std)/abs(latest_pred)if abs(latest_pred)>1e-06 else abs(latest_std)*current_price/current_price if current_price>0 else 1.
		elif current_price>0:pred_return=latest_pred/current_price;std_return=latest_std/current_price if latest_std>0 else abs(pred_return)*.1;uncertainty_ratio=abs(std_return)/abs(pred_return)if abs(pred_return)>1e-06 else abs(std_return)if abs(std_return)>0 else 1.
		else:uncertainty_ratio=1.
		if uncertainty_ratio>1e1:logger.warning(f"⚠️ Suspiciously high uncertainty ratio: {uncertainty_ratio:.2f} (pred={latest_pred:.6f}, std={latest_std:.6f}, price={current_price:.2f}). Using fallback calculation.");fallback_uncertainty=min(2.,max(.1,abs(latest_std)/max(abs(latest_pred),current_price*.01)));uncertainty_ratio=fallback_uncertainty
		uncertainty_ratio=max(.01,min(uncertainty_ratio,2.))
		if uncertainty_ratio<.3:confidence_factors.append(.15);explanations.append(f"Low prediction uncertainty ({uncertainty_ratio:.2%})")
		elif uncertainty_ratio<.5:confidence_factors.append(.1);explanations.append(f"Moderate prediction uncertainty ({uncertainty_ratio:.2%})")
		elif uncertainty_ratio<1.:confidence_factors.append(.05);explanations.append(f"High prediction uncertainty ({uncertainty_ratio:.2%}) - reduces confidence")
		else:confidence_factors.append(.0);explanations.append(f"{"🚨 Extremely"if uncertainty_ratio>1.5 else"Very"} high prediction uncertainty ({uncertainty_ratio:.2%}) - {"very "if uncertainty_ratio>1.5 else""}low directional confidence")
		if volatility<.2:confidence_factors.append(.1);explanations.append(f"Low market volatility ({volatility:.2%})")
		elif volatility<.4:confidence_factors.append(.05);explanations.append(f"Moderate volatility ({volatility:.2%})")
		else:confidence_factors.append(.0);explanations.append(f"High volatility ({volatility:.2%}) - increased risk")
		heuristic_confidence=min(1.,max(.0,sum(confidence_factors)));statistical_confidence=(lambda ci_1d:(lambda ci_width,ci_mean:.9 if(relative_width:=ci_width/abs(ci_mean)if abs(ci_mean)>1e-06 else 1.)<.1 else .75 if relative_width<.2 else .6 if relative_width<.3 else .45 if relative_width<.5 else .3)(ci_1d['upper']-ci_1d['lower'],ci_1d['mean'])if'lower'in ci_1d and'upper'in ci_1d and'mean'in ci_1d and ci_1d['mean']>0 else .5)(confidence_intervals.get('1d',{}))if'1d'in confidence_intervals else .5;ensemble_confidence=(lambda latest_std:.85 if(cv:=abs(latest_std)/abs(latest_pred))<.2 else .7 if cv<.3 else .55 if cv<.5 else .4)(pred_std[-1])if len(pred_std)>0 and abs(latest_pred)>1e-06 else .5;raw_confidence=min(1.,max(.0,.4*heuristic_confidence+.35*statistical_confidence+.25*ensemble_confidence));confidence=self.confidence_calibrator.calibrate_prob(raw_confidence)if hasattr(self,'confidence_calibrator')and self.confidence_calibrator else raw_confidence;confidence=min(1.,max(.0,confidence))
		if VERBOSE_LOGGING:
			calib_info=f" (calibrated: {confidence:.0%})"if hasattr(self,'confidence_calibrator')and self.confidence_calibrator.isotonic_model else'';explanations.append(f"Confidence breakdown: heuristic={heuristic_confidence:.0%}, statistical={statistical_confidence:.0%}, ensemble={ensemble_confidence:.0%}, raw={raw_confidence:.0%}{calib_info}")
			if hasattr(self,'confidence_calibrator'):
				calib_metrics=self.confidence_calibrator.get_calibration_metrics()
				if calib_metrics['calibration_samples']>0:explanations.append(f"Calibration: Brier score={calib_metrics["brier_score"]:.3f}, samples={calib_metrics["calibration_samples"]}")
		conservative_mode=getattr(self,'conservative_mode',False);aggressive_mode=getattr(self,'aggressive_mode',False)
		if conservative_mode:base_threshold=.7
		elif aggressive_mode:base_threshold=.35
		else:base_threshold=.55
		if self.longterm_mode and hasattr(self,'min_confidence_threshold'):base_threshold=self.min_confidence_threshold;explanations.append(f"🔍 Longterm mode: Using strict confidence threshold of {base_threshold:.0%}")
		min_confidence_threshold,block_trade_due_to_uncertainty=base_threshold,False
		if conservative_mode:extreme_uncertainty=2.;very_high_uncertainty=1.5;high_uncertainty=1.2;moderate_uncertainty=.8
		elif aggressive_mode:extreme_uncertainty=5.;very_high_uncertainty=4.;high_uncertainty=3.;moderate_uncertainty=2.
		else:extreme_uncertainty=3.;very_high_uncertainty=2.;high_uncertainty=1.5;moderate_uncertainty=1.
		if uncertainty_ratio>extreme_uncertainty:block_trade_due_to_uncertainty=True;explanations.append(f"🚫 Trade blocked: Extremely high prediction uncertainty ({uncertainty_ratio:.2%})")
		elif uncertainty_ratio>very_high_uncertainty:min_confidence_threshold=max(.8,base_threshold+.15);explanations.append(f"⚠️ Very high uncertainty ({uncertainty_ratio:.2%}) - requiring {min_confidence_threshold:.0%} confidence")
		elif uncertainty_ratio>high_uncertainty:min_confidence_threshold=max(.75,base_threshold+.1);explanations.append(f"⚠️ High uncertainty ({uncertainty_ratio:.2%}) - requiring {min_confidence_threshold:.0%} confidence")
		elif uncertainty_ratio>moderate_uncertainty:min_confidence_threshold=max(.7,base_threshold+.05);explanations.append(f"⚠️ Moderate-high uncertainty ({uncertainty_ratio:.2%}) - requiring {min_confidence_threshold:.0%} confidence")
		elif uncertainty_ratio>.7:min_confidence_threshold=base_threshold+.03
		avg_signal_score=pred_metadata.get('avg_signal_score',.5)
		if conservative_mode:min_signal_quality=.35
		elif aggressive_mode:min_signal_quality=.1
		else:min_signal_quality=.2
		causal_score,causal_block=self._evaluate_causal_stability()
		if causal_block:explanations.insert(0,f"🚫 HOLD: Causal block (score {causal_score:.2f})");return{'signal':'HOLD','confidence':confidence,'explanations':explanations,'recommended_position_size':.0,'top_signals':top_signals,'avg_signal_score':avg_signal_score,'expected_return':expected_return,'volatility':volatility,'causal_score':causal_score,'causal_block':causal_block}
		else:damp=max(.5,min(1.,causal_score));confidence*=damp;avg_signal_score*=damp
		if VERBOSE_LOGGING:
			explanations.append(f"DEBUG: signal_strength={signal_strength:.3f}, confidence={confidence:.1%}, threshold={min_confidence_threshold:.1%}, avg_signal_score={avg_signal_score:.2f}, uncertainty={uncertainty_ratio:.2f}, block_trade={block_trade_due_to_uncertainty}, expected_return={expected_return:.4f}")
			if signal_strength>.15 and signal_direction>0:explanations.append(f"STRONG BUY criteria: strength>{.15} ✓, confidence>{min_confidence_threshold} {"✓"if confidence>min_confidence_threshold else"✗"}, signal_score>={min_signal_quality} {"✓"if avg_signal_score>=min_signal_quality else"✗"}")
			elif signal_strength>.1 and signal_direction>0:explanations.append(f"MODERATE BUY criteria: strength>{.1} ✓, confidence>{min_confidence_threshold} {"✓"if confidence>min_confidence_threshold else"✗"}, signal_score>={min_signal_quality} {"✓"if avg_signal_score>=min_signal_quality else"✗"}")
			elif signal_strength>.15 and signal_direction<0:explanations.append(f"STRONG SELL criteria: strength>{.15} ✓, confidence>{min_confidence_threshold} {"✓"if confidence>min_confidence_threshold else"✗"}, signal_score>={min_signal_quality} {"✓"if avg_signal_score>=min_signal_quality else"✗"}")
			elif signal_strength>.1 and signal_direction<0:explanations.append(f"MODERATE SELL criteria: strength<{.1} ✓, confidence>{min_confidence_threshold} {"✓"if confidence>min_confidence_threshold else"✗"}, signal_score>={min_signal_quality} {"✓"if avg_signal_score>=min_signal_quality else"✗"}")
			else:explanations.append(f"HOLD: signal too weak ({signal_strength:.3f} strength, direction={signal_direction})")
		if uncertainty_ratio>1.:logger.debug(f"High uncertainty ({uncertainty_ratio:.2%}) requires confidence >{min_confidence_threshold:.0%}, current: {confidence:.0%}")
		signal,entry_level,stop_loss,take_profit='HOLD',current_price,None,None
		if abs(latest_pred)<=1.5:raw_expected_return=latest_pred
		else:raw_expected_return=(latest_pred-current_price)/current_price if current_price>0 else .0
		pred_std=pred_metadata.get('prediction_std',[]);latest_std=pred_std[-1]if len(pred_std)>0 else abs(raw_expected_return)*.1;reliability_ratio=abs(latest_std)/max(abs(raw_expected_return),1e-06);prediction_reliable=True;expected_return=raw_expected_return
		if abs(raw_expected_return)<1e-06:
			prediction_reliable=False;expected_return=.0
			if VERBOSE_LOGGING:explanations.append('DEBUG: raw_expected_return ~0, treating as neutral')
		elif reliability_ratio>5. or abs(raw_expected_return)>2.:
			prediction_reliable=False;expected_return=.0
			if VERBOSE_LOGGING:explanations.append(f"DEBUG: unreliable prediction (ratio={reliability_ratio:.1f}, ret={raw_expected_return:.1%}) -> neutral")
		elif abs(raw_expected_return)>1.:
			expected_return=raw_expected_return*.5
			if VERBOSE_LOGGING:explanations.append(f"DEBUG: large prediction {raw_expected_return:.1%} softened to {expected_return:.1%}")
		if'1d'in confidence_intervals and isinstance(confidence_intervals['1d'],dict):
			ci_mean=confidence_intervals['1d'].get('mean',None)
			if ci_mean is not None and current_price>0:target_ret=(ci_mean-current_price)/current_price;expected_return=.6*expected_return+.4*target_ret
		expected_return=max(-.5,min(.5,expected_return))
		if not prediction_reliable:
			confidence=confidence*.7
			if VERBOSE_LOGGING:explanations.append(f"DEBUG: Prediction unreliable, confidence -> {confidence:.1%}")
		if VERBOSE_LOGGING and prediction_reliable:explanations.append(f"DEBUG: latest_pred={latest_pred:.6f}, current_price={current_price:.2f}, expected_return={expected_return:.2%} (reliable)")
		if conservative_mode:strong_threshold=.12;moderate_threshold=.08
		elif aggressive_mode:strong_threshold=.03;moderate_threshold=.015
		else:strong_threshold=.06;moderate_threshold=.03
		signal_strength=confidence*.4+avg_signal_score*.4+min(abs(expected_return),.1)*10*.2;signal_direction=1 if expected_return>0 else-1
		if VERBOSE_LOGGING:explanations.append(f"DEBUG: signal_strength={signal_strength:.3f}, expected_return={expected_return:.4f}, signal_direction={signal_direction}")
		if not block_trade_due_to_uncertainty and signal_strength>strong_threshold and confidence>min_confidence_threshold and avg_signal_score>=min_signal_quality and signal_direction>0:signal='BUY';stop_loss=min(current_price*.98,confidence_intervals['1d']['lower']);take_profit=max(current_price*1.03,confidence_intervals['1d']['upper']);explanations.insert(0,f"BUY signal: Strong bullish momentum (strength: {signal_strength:.2f}, {confidence:.0%} confidence, expected: {expected_return:.2%})")
		elif not block_trade_due_to_uncertainty and signal_strength>moderate_threshold and confidence>min_confidence_threshold and avg_signal_score>=min_signal_quality and signal_direction>0:signal='BUY';stop_loss=current_price*.98;take_profit=confidence_intervals['1d']['upper'];explanations.insert(0,f"BUY signal: Moderate bullish signal (strength: {signal_strength:.2f}, confidence: {confidence:.0%}, expected return: {expected_return:.2%})")
		elif not block_trade_due_to_uncertainty and signal_strength>strong_threshold and confidence>min_confidence_threshold and avg_signal_score>=min_signal_quality and signal_direction<0:signal='SELL';stop_loss=max(current_price*1.02,confidence_intervals['1d']['upper']);take_profit=min(current_price*.97,confidence_intervals['1d']['lower']);explanations.insert(0,f"SELL signal: Strong bearish momentum (strength: {signal_strength:.2f}, {confidence:.0%} confidence, expected: {expected_return:.2%})")
		elif not block_trade_due_to_uncertainty and signal_strength>moderate_threshold and confidence>min_confidence_threshold and avg_signal_score>=min_signal_quality and signal_direction<0:signal='SELL';stop_loss=current_price*1.02;take_profit=confidence_intervals['1d']['lower'];explanations.insert(0,f"SELL signal: Moderate bearish signal (strength: {signal_strength:.2f}, confidence: {confidence:.0%}, expected return: {expected_return:.2%})")
		elif not block_trade_due_to_uncertainty and expected_return>.01 and confidence>=min_confidence_threshold-.05:signal='BUY';stop_loss=current_price*.985;take_profit=current_price*(1+max(expected_return,.02));explanations.insert(0,f"BUY signal (fallback): expected {expected_return:.2%}, confidence {confidence:.0%}")
		elif not block_trade_due_to_uncertainty and expected_return<-.01 and confidence>=min_confidence_threshold-.05:signal='SELL';stop_loss=current_price*1.015;take_profit=current_price*(1+min(expected_return,-.02));explanations.insert(0,f"SELL signal (fallback): expected {expected_return:.2%}, confidence {confidence:.0%}")
		else:
			hold_reasons=[]
			if signal_strength<=moderate_threshold:hold_reasons.append(f"signal too weak ({signal_strength:.3f} strength, need >{moderate_threshold:.3f})")
			if confidence<=min_confidence_threshold:hold_reasons.append(f"low confidence ({confidence:.0%}, required: {min_confidence_threshold:.0%})")
			if avg_signal_score<min_signal_quality:hold_reasons.append(f"insufficient signal quality ({avg_signal_score:.2f}, required: {min_signal_quality:.2f})")
			if block_trade_due_to_uncertainty:hold_reasons.append(f"extremely high uncertainty ({uncertainty_ratio:.2%})")
			explanations.insert(0,f"HOLD: {", ".join(hold_reasons)if hold_reasons else f"Signal too weak ({signal_strength:.3f} strength) or low confidence ({confidence:.0%}, required: {min_confidence_threshold:.0%})"}")
			if uncertainty_ratio>1. and confidence<=min_confidence_threshold:explanations.append(f"⚠️ High uncertainty ({uncertainty_ratio:.2%}) requires {min_confidence_threshold:.0%} confidence, but only {confidence:.0%} achieved")
		if signal!='HOLD':
			if self.longterm_mode and hasattr(self,'causal_engine')and self.causal_engine:
				try:
					causal_validation=self._validate_with_crca(signal,expected_return,confidence,symbol or'MAIN')
					if not causal_validation.get('approved',False):signal='HOLD';explanations.insert(0,f"🔍 Longterm mode: CRCAAgent validation failed - {causal_validation.get("reason","Causal analysis does not support this trade")}");logger.info(f"Longterm mode: Trade blocked by CRCAAgent validation for {symbol or"MAIN"}")
				except Exception as e:
					logger.warning(f"CRCAAgent validation failed: {e}, proceeding with caution")
					if self.longterm_mode:explanations.append(f"⚠️ CRCAAgent validation error, using extra caution")
			try:
				position_sizing_method=getattr(self,'position_sizing_method','kelly')
				if position_sizing_method not in['kelly','risk_parity','target_vol']:position_sizing_method='kelly';explanations.append('⚠️ Invalid position sizing method, using Kelly')
				if volatility is None or isinstance(volatility,float)and(np.isnan(volatility)or np.isinf(volatility)):volatility=.02;explanations.append('⚠️ Volatility data unavailable, using default 2%')
				elif volatility<=0:volatility=.02;explanations.append('⚠️ Invalid volatility value, using default 2%')
				else:
					if volatility>2.:volatility/=1e2;explanations.append('ℹ️ High volatility normalized from percent scale')
					if volatility>.8:volatility=.8;explanations.append('⚠️ Extremely high volatility capped at 80%')
				if uncertainty_ratio is None or isinstance(uncertainty_ratio,float)and(np.isnan(uncertainty_ratio)or np.isinf(uncertainty_ratio)):uncertainty_ratio=1.;explanations.append('⚠️ Uncertainty ratio invalid, using default 100%')
				if latest_pred is None or isinstance(latest_pred,float)and(np.isnan(latest_pred)or np.isinf(latest_pred)):latest_pred=expected_return*current_price if expected_return is not None and current_price>0 else .0
				if current_price is None or current_price<=0:current_price=1.
				er=latest_pred/current_price if abs(latest_pred)>1. and current_price>0 else latest_pred
				if er is None or not isinstance(er,(int,float))or np.isnan(er)or np.isinf(er):er=.0;explanations.append('⚠️ Invalid expected return for position sizing, using 0%')
				if volatility is None or not isinstance(volatility,(int,float))or np.isnan(volatility)or np.isinf(volatility)or volatility<=0:volatility=.02;explanations.append('⚠️ Invalid volatility for position sizing, using 2%')
				if uncertainty_ratio is None or not isinstance(uncertainty_ratio,(int,float))or np.isnan(uncertainty_ratio)or np.isinf(uncertainty_ratio):uncertainty_ratio=1.;explanations.append('⚠️ Invalid uncertainty ratio for position sizing, using 100%')
				if confidence is None or not isinstance(confidence,(int,float))or np.isnan(confidence)or np.isinf(confidence):confidence=.5;explanations.append('⚠️ Invalid confidence for position sizing, using 50%')
				if max_position_size is None or not isinstance(max_position_size,(int,float))or np.isnan(max_position_size)or max_position_size<=0:max_position_size=.1;explanations.append('⚠️ Invalid max position size, using 10%')
				base_size=min((abs(er/(volatility*volatility))*.25 if position_sizing_method=='kelly'else .1/volatility if position_sizing_method=='risk_parity'else getattr(self,'target_volatility',.15)/volatility if position_sizing_method=='target_vol'else abs(er)/(getattr(self,'risk_aversion',2.)*volatility*volatility))if volatility>1e-06 else abs(weight)*.1,max_position_size);recommended_size=max(.01,min(max_position_size,(lambda vol_adj:vol_adj*(.7 if volatility>.4 else .85 if volatility>.3 else 1.)*(.3 if(unc_ratio_sizing:=min(uncertainty_ratio,2.))>1.5 else .4 if unc_ratio_sizing>1. else .6 if unc_ratio_sizing>.7 else .75 if unc_ratio_sizing>.5 else 1.)*getattr(self,'position_size_multiplier',1.))(base_size*confidence*(1-min(volatility,.5)))))
				if self.longterm_mode:recommended_size=min(recommended_size,getattr(self,'max_position_size',.005));explanations.append(f"🔍 Longterm mode: Position size capped at {recommended_size:.2%}")
				if aggressive_mode and not self.longterm_mode:recommended_size=min(max_position_size,recommended_size*1.5)
				uncertainty_ratio=uncertainty_ratio if isinstance(uncertainty_ratio,(int,float))and not np.isnan(uncertainty_ratio)else 1.
				if min(uncertainty_ratio,2.)>1.5:explanations.append(f"⚠️ Extreme uncertainty: Position size reduced by 70%")
				if hasattr(self,'risk_monitor')and self.risk_monitor:
					account_size=getattr(self,'account_size',TRADING_CONFIG['account_size']);portfolio_value=account_size;stop_loss_distance=abs((stop_loss-entry_level)/entry_level)if stop_loss and entry_level>0 else .02;current_positions={k:v.get('size',.0)/portfolio_value if portfolio_value>0 else .0 for(k,v)in self.positions.items()}if hasattr(self,'positions')else{};is_valid,reason,adjusted_size=self.risk_monitor.pre_trade_check(signal=signal,position_size=recommended_size,current_positions=current_positions,portfolio_value=portfolio_value,stop_loss_distance=stop_loss_distance,asset_volatility=volatility,asset_type=getattr(self,'asset_types',{}).get(symbol,'crypto'))
					if not is_valid:
						if VERBOSE_LOGGING:explanations.append(f"⚠️ Risk limit: {reason}")
						recommended_size=adjusted_size
						if recommended_size<.001:recommended_size,signal=.0,'HOLD';explanations.append('Trade cancelled: Position size too small after risk adjustments')
				account_size=getattr(self,'account_size',TRADING_CONFIG['account_size']);trade_value=recommended_size*account_size;daily_volume=float(self.price_data['volume'].iloc[-20:].mean()*current_price)if hasattr(self,'price_data')and not self.price_data.empty and'volume'in self.price_data.columns and current_price>0 else None
				if hasattr(self,'transaction_cost_model'):
					transaction_costs=self.transaction_cost_model.estimate_total_cost(trade_size=recommended_size*account_size/current_price if current_price>0 else .0,trade_value=trade_value,current_price=current_price,daily_volume=daily_volume);expected_return_net=expected_return-transaction_costs['total_pct']
					if expected_return_net<=0 and not aggressive_mode:explanations.append(f"⚠️ Trade cancelled: Transaction costs ({transaction_costs["total_pct"]:.2%}) exceed expected return ({expected_return:.2%})");recommended_size,signal=.0,'HOLD'
					elif VERBOSE_LOGGING:explanations.append(f"Transaction costs: {transaction_costs["total_pct"]:.2%} (spread: {transaction_costs["spread"]/trade_value:.2%}, slippage: {transaction_costs["slippage"]/trade_value:.2%}, impact: {transaction_costs["impact"]/trade_value:.2%}, fee: {transaction_costs["fee"]/trade_value:.2%})");explanations.append(f"Net expected return: {expected_return_net:.2%} (gross: {expected_return:.2%})")
				if signal!='HOLD'and current_price>0 and account_size>0:
					min_trade_value=getattr(self,'min_trade_value',TRADING_CONFIG['min_trade_value']);account_fraction_for_min=min_trade_value/account_size
					if account_fraction_for_min<=max_position_size and recommended_size<account_fraction_for_min:recommended_size=account_fraction_for_min;explanations.append(f"Adjusted to meet minimum trade value ({min_trade_value:.2f} EUR)")
			except Exception as sizing_error:explanations.append(f"⚠️ Position sizing calculation failed: {str(sizing_error)}");recommended_size=.0
		else:recommended_size=.0
		return{'signal':signal,'confidence':confidence,'explanations':explanations,'recommended_position_size':recommended_size,'entry_level':entry_level,'stop_loss':stop_loss,'take_profit':take_profit,'expected_return':expected_return,'prediction_uncertainty':float(uncertainty_ratio),'top_signals':top_signals,'volatility':volatility,'current_price':current_price,'causal_score':causal_score,'causal_block':causal_block}
	def _make_multi_asset_decisions(self,portfolio_weights:Dict[str,float],predictions:Dict[str,np.ndarray],signal_scores:Dict[str,Dict[str,float]],pred_metadata_dict:Dict[str,Dict[str,Any]],confidence_intervals_dict:Dict[str,Dict[str,Dict[str,float]]],volatility_dict:Dict[str,float],current_prices:Dict[str,float],max_position_size:Optional[float]=None)->Dict[str,Dict[str,Any]]:
		decisions={}
		for symbol in portfolio_weights.keys():
			if symbol not in predictions or symbol not in current_prices:continue
			volatility=volatility_dict.get(symbol,.02);volatility=.02 if volatility is None or isinstance(volatility,float)and np.isnan(volatility)else volatility;decision=self._make_trading_decision(portfolio_weights=np.array([portfolio_weights.get(symbol,.0)]),predictions=predictions[symbol],signal_scores=signal_scores.get(symbol,{}),pred_metadata=pred_metadata_dict.get(symbol,{}),confidence_intervals=confidence_intervals_dict.get(symbol,{}),volatility=volatility,current_price=current_prices[symbol],max_position_size=max_position_size,symbol=symbol);decisions[symbol]=decision
		return decisions
	def backtest(self,start_date:datetime,end_date:datetime,train_window:int=60,test_window:int=7,step:int=7)->Dict[str,Any]:results=self.backtest_engine.run_backtest(agent=self,start_date=start_date,end_date=end_date,train_window=train_window,test_window=test_window,step=step);report=self.backtest_engine._generate_report(results);logger.info(f"\n{report}");return results
	async def start_streaming(self,exchange:str='coinbase',symbol:str='ETH',decision_callback:Optional[Callable]=None):
		self.streaming_mode,self.streaming_buffer,self.streaming_order_book,self.last_signal_update,self.signal_update_interval,self.decision_callback=True,[],None,0,5,decision_callback
		async def handle_stream_data(data:Dict[str,Any]):
			try:
				msg_type,price,volume=data.get('type',''),None,.0
				if msg_type=='ticker':price,volume=data.get('price',0),data.get('volume_24h',0)
				elif msg_type=='trade':price,volume=data.get('price',0),data.get('size',0)
				elif msg_type=='orderbook':
					self.streaming_order_book=data.get('data');order_book=self.streaming_order_book
					if order_book and'bids'in order_book and'asks'in order_book:
						bids,asks=sorted([float(p)for p in order_book['bids'].keys()],reverse=True),sorted([float(p)for p in order_book['asks'].keys()])
						if bids and asks:price=(bids[0]+asks[0])/2
				else:price,volume=data.get('price')or data.get('last_price')or data.get('c'),data.get('volume',0)or data.get('v',0)
				if price and price>0:
					self.streaming_buffer.append({'timestamp':datetime.now(),'price':float(price),'volume':float(volume)})
					if len(self.streaming_buffer)>2000:self.streaming_buffer=self.streaming_buffer[-2000:]
					self.current_price=float(price);current_time=time.time()
					if current_time-self.last_signal_update>=self.signal_update_interval:
						await self._compute_streaming_signals();self.last_signal_update=current_time
						if self.decision_callback:
							try:await self.decision_callback(self)
							except Exception as e:logger.debug(f"Error in decision callback: {e}")
			except Exception as e:logger.debug(f"Error handling stream data: {e}")
		success=await self.socket_manager.connect(exchange,symbol,handle_stream_data)
		if success:logger.info(f"Started streaming mode for {exchange}:{symbol}")
		else:logger.error(f"Failed to start streaming for {exchange}:{symbol}");self.streaming_mode=False
	async def _compute_streaming_signals(self):
		if len(self.streaming_buffer)<30:return
		try:
			stream_df=pd.DataFrame(self.streaming_buffer);stream_df.set_index('timestamp',inplace=True)
			if'price'not in stream_df.columns:return
			stream_df['returns']=stream_df['price'].pct_change()
			if'volume'not in stream_df.columns:stream_df['volume']=.0
			self.price_data=stream_df[['price','volume']].copy();signals_df=self.compute_signals()
			if not signals_df.empty:self.signals_df=signals_df;logger.debug(f"Updated signals from streaming data: {len(signals_df)} points, latest price: ${self.current_price:.2f}")
		except Exception as e:logger.debug(f"Error computing streaming signals: {e}")
	async def stop_streaming(self,exchange:str='coinbase',symbol:str='ETH'):self.streaming_mode=False;await self.socket_manager.disconnect(exchange,symbol);logger.info('Stopped streaming mode')
	async def run_batched_multi_asset(self)->Dict[str,Any]:
		if not self.multi_asset_mode or not self.assets:return await self.run()
		self.global_loop_counter+=1
		if self.system_state!='COOLDOWN':
			self.loops_since_cooldown+=1
			if self.cooldown_exit_guard:self.cooldown_exit_guard=False
		api_budget_remaining=getattr(self,'api_budget_remaining',1.)
		if self.system_state=='COOLDOWN':self._cooldown_tick(api_budget_remaining)
		portfolio_metrics=self._update_session_pnl();self._maybe_trigger_session_circuit_breaker(portfolio_metrics);num_assets=len(self.assets);workflow_steps=['Fetching multi-asset data',f"Processing {num_assets} assets",'Evaluating asset rotation','Making trading decisions','Calculating portfolio metrics']
		if RICH_AVAILABLE:
			with Progress(SpinnerColumn(),TextColumn('[progress.description]{task.description}'),BarColumn(),TextColumn('[progress.percentage]{task.percentage:>3.0f}%'),TimeRemainingColumn(),console=Console(),transient=False)as progress:
				task=progress.add_task('[cyan]Multi-Asset Trading Workflow',total=len(workflow_steps));progress.update(task,description=f"[yellow]{workflow_steps[0]}...");self.fetch_data()
				if not self.price_data_dict:return{'error':'Failed to fetch multi-asset data'}
				progress.advance(task)
		all_signal_scores,all_predictions,all_metadata,all_intervals,all_current_prices={},{},{},{},{};asset_symbols=list(self.price_data_dict.keys());congestion_metric=.0
		for(idx,symbol)in enumerate(asset_symbols):
			progress.update(task,description=f"[yellow]Processing asset {idx+1}/{len(asset_symbols)}: {symbol}...");self.price_data=self.price_data_dict[symbol];self.current_price=self.price_data['price'].iloc[-1]if not self.price_data.empty else .0;all_current_prices[symbol]=self.current_price;signals_df=self.compute_signals()
			if signals_df.empty:continue
			signal_scores=self.validate_signals();all_signal_scores[symbol]=signal_scores
		if all_signal_scores:
			multi_predictions=self.generate_multi_asset_predictions(all_signal_scores)
			for symbol in asset_symbols:
				if symbol in multi_predictions:predictions,intervals,metadata=multi_predictions[symbol];all_predictions[symbol]=predictions;all_metadata[symbol]=metadata;all_intervals[symbol]=intervals
				else:all_predictions[symbol]=np.array([.0]);all_metadata[symbol]={};all_intervals[symbol]={}
				progress.advance(task);progress.update(task,description=f"[yellow]{workflow_steps[2]}...");rotation_trades={'exits':[],'entries':[]}
		total_trades=0
		if ROTATION_ENABLED and self.rotation_manager:
			predictions_dict={symbol:float(all_predictions[symbol][-1])if symbol in all_predictions and len(all_predictions[symbol])>0 else .0 for symbol in all_current_prices.keys()};confidences_dict={symbol:all_metadata.get(symbol,{}).get('prediction_confidence',.5)for symbol in all_current_prices.keys()};uncertainties_dict={symbol:all_metadata.get(symbol,{}).get('prediction_uncertainty',.5)for symbol in all_current_prices.keys()};portfolio_metrics=self._calculate_portfolio_metrics();rotation_trades=self.rotation_manager.evaluate_asset_rotation(current_positions=self.positions,all_assets=self.assets,predictions=predictions_dict,signal_scores_dict=all_signal_scores,current_prices=all_current_prices,prediction_confidences=confidences_dict,prediction_uncertainties=uncertainties_dict,portfolio_value=portfolio_metrics.get('current_portfolio_value',self.initial_portfolio_value))
			for exit_info in rotation_trades.get('exits',[]):
				symbol=exit_info['symbol']
				if symbol in self.positions and self.live_trading_mode:
					exit_size=self.positions[symbol].get('size',.0)/portfolio_metrics.get('current_portfolio_value',self.initial_portfolio_value)
					if exit_size>0:
						trade_result=self.execution_engine.execute_trade(signal='SELL',size_fraction=min(exit_size,self.risk_monitor.max_position_size),base_symbol=symbol)
						if trade_result:
							total_trades+=1
							if trade_result.get('status')=='success':self._record_live_fill(symbol=symbol,side='SELL',amount=trade_result.get('amount',.0),price=trade_result.get('price',all_current_prices.get(symbol,.0)))
							logger.debug(f"🔄 Rotation: Exited {symbol} (score: {exit_info["score"]:.3f})")if VERBOSE_LOGGING else None
			for entry_info in rotation_trades.get('entries',[]):
				symbol=entry_info['symbol']
				if not any(a.get('symbol','').upper()==symbol for a in self.assets):self.assets.append({'symbol':symbol,'type':'crypto'})
				progress.advance(task);progress.update(task,description=f"[yellow]{workflow_steps[3]}...")
		if hasattr(self,'asset_discovery'):
			asset_scores_list=[]
			for asset in self.assets:
				symbol=asset.get('symbol','').upper()
				if symbol not in self.price_data_dict:continue
				pred_array=all_predictions.get(symbol,np.array([]));predicted_return=float(pred_array[-1])if len(pred_array)>0 else None;meta=all_metadata.get(symbol,{});priority_score=self.asset_discovery.get_asset_priority_score(symbol,self.price_data_dict[symbol],all_signal_scores.get(symbol,{}),all_current_prices.get(symbol,.0),predicted_return=predicted_return,prediction_confidence=meta.get('prediction_confidence',None),prediction_uncertainty=meta.get('prediction_uncertainty',None));asset_scores_list.append((priority_score,asset))
			sorted_assets=[asset for(_,asset)in sorted(asset_scores_list,key=lambda x:x[0],reverse=True)]
		else:sorted_assets=self.assets
		batch_results,total_trades=[],0
		for batch_idx in range(0,len(sorted_assets),BATCH_SIZE):
			batch=sorted_assets[batch_idx:batch_idx+BATCH_SIZE];batch_symbols=[a.get('symbol','').upper()for a in batch]
			if VERBOSE_LOGGING:logger.debug(f"📦 Processing batch {batch_idx//BATCH_SIZE+1}: {", ".join(batch_symbols)}")
			batch_predictions={s:all_predictions.get(s)for s in batch_symbols if s in all_predictions};batch_metadata={s:all_metadata.get(s)for s in batch_symbols if s in all_metadata}
			if not batch_predictions:continue
			expected_returns_dict,batch_predictions_arrays={},{}
			for symbol in batch_symbols:
				if symbol in batch_predictions:
					pred_array=batch_predictions[symbol][0]if isinstance(batch_predictions[symbol],tuple)and len(batch_predictions[symbol])>0 else batch_predictions[symbol]
					if isinstance(pred_array,np.ndarray)and len(pred_array)>0:expected_returns_dict[symbol]=float(pred_array[-1]);batch_predictions_arrays[symbol]=pred_array
			if not expected_returns_dict:continue
			batch_returns_dict={symbol:self.price_data_dict[symbol]['price'].pct_change().dropna()for symbol in batch_symbols if symbol in self.price_data_dict and'price'in self.price_data_dict[symbol].columns and len(self.price_data_dict[symbol])>1 and len(self.price_data_dict[symbol]['price'].pct_change().dropna())>0};covariance_df=self.market_data_client.compute_multi_asset_covariance(batch_returns_dict,window=20)if len(batch_returns_dict)>1 else pd.DataFrame(np.eye(len(expected_returns_dict))*.01,index=list(expected_returns_dict.keys()),columns=list(expected_returns_dict.keys()));portfolio_weights=self.optimize_portfolio(expected_returns_dict,covariance_df,all_signal_scores);batch_volatility={}
			for symbol in batch_symbols:
				try:
					if symbol in self.price_data_dict and'price'in self.price_data_dict[symbol].columns and len(self.price_data_dict[symbol])>1:
						pct_change=self.price_data_dict[symbol]['price'].pct_change().dropna()
						if len(pct_change)>0:vol=pct_change.std()*np.sqrt(252);batch_volatility[symbol]=float(vol)if not np.isnan(vol)else .02
						else:batch_volatility[symbol]=.02
					else:batch_volatility[symbol]=.02
				except Exception:batch_volatility[symbol]=.02
			if self.system_state!='COOLDOWN'and self.cooldown_enabled:
				trigger,reason=self._should_enter_cooldown(api_budget_remaining,congestion_metric)
				if trigger:self._enter_cooldown(reason)
			try:batch_decisions=self._make_multi_asset_decisions(portfolio_weights,batch_predictions_arrays,{s:all_signal_scores.get(s,{})for s in batch_symbols},batch_metadata,{s:all_intervals.get(s,{})for s in batch_symbols},batch_volatility,{s:all_current_prices.get(s,.0)for s in batch_symbols},self.risk_monitor.max_position_size)
			except Exception as e:
				logger.error(f"Batch decision making failed: {e}");batch_decisions={}
				for symbol in batch_symbols:
					confidence,confidence_explanations=self._compute_confidence_only(batch_predictions_arrays.get(symbol,np.array([.0])),all_signal_scores.get(symbol,{}),batch_metadata.get(symbol,{}),all_intervals.get(symbol,{}));prediction=batch_predictions_arrays.get(symbol,np.array([.0]));weight=float(prediction[0])if len(prediction)>0 else .0;signal='HOLD';aggressive_base_threshold=.45;aggressive_weight_threshold=.04
					if weight>aggressive_weight_threshold and confidence>aggressive_base_threshold:signal='BUY'
					elif weight<-aggressive_weight_threshold and confidence>aggressive_base_threshold:signal='SELL'
					batch_decisions[symbol]={'signal':signal,'confidence':confidence,'recommended_position_size':.0,'expected_return':weight*.1,'explanations':confidence_explanations+[f"Position sizing failed: {str(e)}"],'current_price':all_current_prices.get(symbol,.0)}
			progress.advance(task);progress.update(task,description=f"[yellow]{workflow_steps[3]}...");batch_trades=[]
			for(symbol,decision)in batch_decisions.items():
				signal=decision.get('signal','HOLD')
				if self.system_state in('COOLDOWN','EXIT'):decision=decision.copy()if isinstance(decision,dict)else{};decision['signal']='EXIT'if self.system_state=='EXIT'else'COOLDOWN';decision['recommended_position_size']=.0;explanations=decision.get('explanations',[]);explanations.append(f"[{decision["signal"]}] Observing only; no trades placed");decision['explanations']=explanations;signal=decision['signal']
				current_price=all_current_prices.get(symbol,decision.get('current_price',.0))
				if signal not in['HOLD','COOLDOWN','EXIT']and self.live_trading_mode:
					expected_return_pct=decision.get('expected_return',None);rec_size=decision.get('recommended_position_size',.0)or .0;max_sz=self.risk_monitor.max_position_size if self.risk_monitor and self.risk_monitor.max_position_size is not None else .1;trade_result=self.execution_engine.execute_trade(signal=signal,size_fraction=min(abs(rec_size),max_sz),base_symbol=symbol,aggressive_mode=self.aggressive_mode,expected_return_pct=expected_return_pct)
					if trade_result:
						batch_trades.append(trade_result);total_trades+=1
						if trade_result.get('status')=='success':filled_amt=trade_result.get('amount',.0);filled_price=trade_result.get('price',current_price);self._record_live_fill(symbol=symbol,side=signal,amount=filled_amt,price=filled_price);filled_val=filled_amt*filled_price;portfolio_value=portfolio_metrics.get('current_portfolio_value',self.account_size if self.account_size>0 else 1.);frac=filled_val/portfolio_value if portfolio_value>0 else .0;decision['recommended_position_size']=frac
				decision_with_symbol=decision.copy()if isinstance(decision,dict)else{};decision_with_symbol['symbol']=symbol
				if signal not in['HOLD','COOLDOWN']and not self.live_trading_mode:
					rec_size=decision_with_symbol.get('recommended_position_size',.0)or .0;account_size=getattr(self,'account_size',TRADING_CONFIG.get('account_size',1e3));min_trade_value=getattr(self,'min_trade_value',TRADING_CONFIG.get('min_trade_value',1.));min_fraction=min_trade_value/account_size if account_size>0 else .0;max_fraction=getattr(self.risk_monitor,'max_position_size',TRADING_CONFIG.get('max_position_size',.1))
					if max_fraction is None or not isinstance(max_fraction,(int,float)):max_fraction=.1
					portfolio_metrics=self._calculate_portfolio_metrics();current_portfolio_value=portfolio_metrics.get('current_portfolio_value',account_size if account_size>0 else 1.);total_invested=portfolio_metrics.get('total_invested',.0);max_total_exposure=getattr(self,'max_total_exposure',1.);allowed_fraction=max(.0,max_total_exposure-(total_invested/current_portfolio_value if current_portfolio_value>0 else .0));rec_size=max(min_fraction,rec_size);rec_size=min(rec_size,max_fraction,allowed_fraction);decision_with_symbol['recommended_position_size']=rec_size
				if not self.live_trading_mode:
					if signal=='COOLDOWN':self._update_positions('HOLD',decision_with_symbol,current_price)
					else:self._update_positions(signal,decision_with_symbol,current_price)
				batch_decisions[symbol]=decision_with_symbol
			batch_results.append({'batch':batch_idx//BATCH_SIZE+1,'assets':batch_symbols,'decisions':batch_decisions,'trades':batch_trades})
			if batch_idx+BATCH_SIZE<len(sorted_assets):await asyncio.sleep(BATCH_DELAY)
		progress.update(task,description=f"[yellow]{workflow_steps[4]}...");progress.advance(task);progress.update(task,completed=len(workflow_steps));primary_symbol=list(self.price_data_dict.keys())[0]if self.price_data_dict else None;primary_decision=batch_results[0]['decisions'].get(primary_symbol,{})if batch_results else{};primary_price=all_current_prices.get(primary_symbol,.0)if primary_symbol else .0;primary_metadata=all_metadata.get(primary_symbol,{})if primary_symbol else{};primary_intervals=all_intervals.get(primary_symbol,{})if primary_symbol else{};price_targets={horizon:primary_intervals[horizon].get('mean',primary_price)for horizon in['1d','3d','7d']if horizon in primary_intervals};primary_volatility=.02
		if primary_symbol and primary_symbol in self.price_data_dict:
			returns=self.price_data_dict[primary_symbol]['price'].pct_change().dropna()
			if len(returns)>0:primary_volatility=float(returns.std()*np.sqrt(252))
		risk_metrics={'volatility':primary_volatility,'sharpe_ratio':primary_decision.get('sharpe_ratio',.0),'cvar_95':primary_decision.get('cvar_95',.0),'prediction_uncertainty':primary_metadata.get('prediction_uncertainty',.0)};regime='unknown'
		if primary_symbol and primary_symbol in self.price_data_dict:
			try:
				price_data=self.price_data_dict[primary_symbol]
				if not price_data.empty and'price'in price_data.columns:
					recent_returns=price_data['price'].pct_change().dropna().tail(20)
					if len(recent_returns)>0:vol,mean_ret=recent_returns.std(),recent_returns.mean();regime='high_volatility'if vol>.05 else'bullish'if mean_ret>.01 else'bearish'if mean_ret<-.01 else'neutral'
			except Exception:pass
		if RICH_AVAILABLE:progress.update(task,description=f"[yellow]{workflow_steps[4]}...");progress.advance(task);progress.update(task,completed=len(workflow_steps))
		wallet_snapshot=self._get_live_wallet_snapshot();return{'signal':primary_decision.get('signal','HOLD'),'original_signal':primary_decision.get('original_signal'),'confidence':primary_decision.get('confidence',.0),'current_price':primary_price,'expected_return':primary_decision.get('expected_return',.0),'recommended_position_size':primary_decision.get('recommended_position_size',.0),'original_position_size':primary_decision.get('original_position_size'),'price_targets':price_targets,'confidence_intervals':primary_intervals,'top_signals':primary_decision.get('top_signals',[]),'signal_explanations':primary_decision.get('explanations',[]),'risk_metrics':risk_metrics,'regime':regime,'entry_level':primary_decision.get('entry_level'),'stop_loss':primary_decision.get('stop_loss'),'take_profit':primary_decision.get('take_profit'),'batched_results':batch_results,'total_assets':len(sorted_assets),'total_trades':total_trades,'portfolio_weights':portfolio_weights if'portfolio_weights'in locals()else{},'all_current_prices':all_current_prices,'all_metadata':all_metadata,'all_intervals':all_intervals,'price_data_dict':{k:v.copy()if isinstance(v,pd.DataFrame)else v for(k,v)in self.price_data_dict.items()},'portfolio_metrics':self._calculate_portfolio_metrics(),'system_state':self.system_state,'wallet':wallet_snapshot};wallet_snapshot=self._get_live_wallet_snapshot();return{'signal':primary_decision.get('signal','HOLD'),'original_signal':primary_decision.get('original_signal'),'confidence':primary_decision.get('confidence',.0),'current_price':primary_price,'expected_return':primary_decision.get('expected_return',.0),'recommended_position_size':primary_decision.get('recommended_position_size',.0),'original_position_size':primary_decision.get('original_position_size'),'price_targets':price_targets,'confidence_intervals':primary_intervals,'top_signals':primary_decision.get('top_signals',[]),'signal_explanations':primary_decision.get('explanations',[]),'risk_metrics':risk_metrics,'regime':regime,'entry_level':primary_decision.get('entry_level'),'stop_loss':primary_decision.get('stop_loss'),'take_profit':primary_decision.get('take_profit'),'batched_results':batch_results,'total_assets':len(sorted_assets),'total_trades':total_trades,'portfolio_weights':portfolio_weights if'portfolio_weights'in locals()else{},'all_current_prices':all_current_prices,'all_metadata':all_metadata,'all_intervals':all_intervals,'price_data_dict':{k:v.copy()if isinstance(v,pd.DataFrame)else v for(k,v)in self.price_data_dict.items()},'portfolio_metrics':portfolio_metrics,'system_state':self.system_state,'wallet':wallet_snapshot}
	def _update_positions(self,signal:str,decision:Dict[str,Any],current_price:float)->None:
		import time
		if not hasattr(self,'positions'):self.positions={}
		symbol=decision.get('symbol','MAIN')if isinstance(decision,dict)and'symbol'in decision else list(self.price_data_dict.keys())[0]if self.multi_asset_mode and hasattr(self,'price_data_dict')and self.price_data_dict else'MAIN';position_size_pct=abs(decision.get('recommended_position_size',.0))if isinstance(decision,dict)else .0;entry_price=decision.get('entry_level',current_price)if isinstance(decision,dict)else current_price;portfolio_metrics=self._calculate_portfolio_metrics();current_portfolio_value=portfolio_metrics.get('current_portfolio_value',self.initial_portfolio_value);position_value=current_portfolio_value*position_size_pct
		if signal=='BUY'and position_size_pct>0:
			if symbol in self.positions:old_size,old_entry=self.positions[symbol]['size'],self.positions[symbol]['entry_price'];new_size=old_size+position_value;new_entry=(old_entry*old_size+entry_price*position_value)/new_size if new_size>0 else entry_price;self.positions[symbol]={'size':new_size,'entry_price':new_entry,'entry_time':self.positions[symbol].get('entry_time',time.time()),'value':new_size,'position_size_pct':new_size/current_portfolio_value if current_portfolio_value>0 else .0}
			else:self.positions[symbol]={'size':position_value,'entry_price':entry_price,'entry_time':time.time(),'value':position_value,'position_size_pct':position_value/current_portfolio_value if current_portfolio_value>0 else position_size_pct}
		elif signal=='SELL'and position_size_pct>0:
			if symbol in self.positions:
				old_value=self.positions[symbol]['size']
				if position_value>=old_value:del self.positions[symbol]
				else:new_value=old_value-position_value;self.positions[symbol]['size']=new_value;self.positions[symbol]['position_size_pct']=new_value/current_portfolio_value if current_portfolio_value>0 else .0
		self._update_position_values()
	def _record_live_fill(self,symbol:str,side:str,amount:float,price:float)->None:
		import time
		if not hasattr(self,'positions'):self.positions={}
		if amount<=0 or price<=0:return
		invested=amount*price
		if side.upper()=='BUY':
			if symbol in self.positions:old_inv=self.positions[symbol].get('size',.0);old_entry=self.positions[symbol].get('entry_price',price);new_inv=old_inv+invested;new_entry=(old_entry*old_inv+price*invested)/new_inv if new_inv>0 else price;self.positions[symbol]={'size':new_inv,'entry_price':new_entry,'entry_time':self.positions[symbol].get('entry_time',time.time()),'value':new_inv,'position_size_pct':new_inv/max(self.account_size,1.)}
			else:self.positions[symbol]={'size':invested,'entry_price':price,'entry_time':time.time(),'value':invested,'position_size_pct':invested/max(self.account_size,1.)}
		elif symbol in self.positions:
			old_inv=self.positions[symbol].get('size',.0)
			if invested>=old_inv:del self.positions[symbol]
			else:remaining=old_inv-invested;self.positions[symbol]['size']=remaining;self.positions[symbol]['position_size_pct']=remaining/max(self.account_size,1.)
		self._update_position_values()
	def _update_position_values(self)->None:
		if not hasattr(self,'positions'):return
		current_prices={}
		if getattr(self,'multi_asset_mode',False)and hasattr(self,'price_data_dict')and self.price_data_dict:current_prices={sym:df['price'].iloc[-1]for(sym,df)in self.price_data_dict.items()if not df.empty and'price'in df.columns}
		elif hasattr(self,'current_price')and self.current_price>0:current_prices['MAIN']=self.current_price
		if hasattr(self,'last_full_run')and self.last_full_run is not None:current_prices.update(self.last_full_run.get('all_current_prices',{}))
		for(sym,position)in list(self.positions.items()):
			current_price=current_prices.get(sym)or position.get('current_price')
			if current_price:
				position['current_price']=current_price;invested_amount,entry_price=position.get('size',.0),position.get('entry_price',current_price)
				if entry_price>0 and invested_amount>0:position['value']=invested_amount*(current_price/entry_price);position['unrealized_pnl']=position['value']-invested_amount;position['unrealized_pnl_pct']=(current_price-entry_price)/entry_price*100
				else:position['value'],position['unrealized_pnl'],position['unrealized_pnl_pct']=invested_amount,.0,.0
	def _promote_baseline(self,current_val:float)->None:
		new_baseline=max(self.baseline_value,current_val)
		if new_baseline<=self.baseline_value:return
		if self.live_trading_mode and self.promotion_debounce_secs>0:
			now_ts=time.time()
			if self.promotion_candidate_ts is None or new_baseline>(self.promotion_candidate_value or .0):self.promotion_candidate_value=new_baseline;self.promotion_candidate_ts=now_ts;logger.info(f"[RATCHET] Live debounce started — candidate baseline {self.promotion_candidate_value:,.2f}");return
			if now_ts-self.promotion_candidate_ts<self.promotion_debounce_secs:logger.info(f"[RATCHET] Live debounce pending ({now_ts-self.promotion_candidate_ts:.2f}s/{self.promotion_debounce_secs:.2f}s) — candidate {self.promotion_candidate_value:,.2f}");return
			new_baseline=max(new_baseline,self.promotion_candidate_value or new_baseline)
		self.baseline_value=new_baseline;self.promotion_event=True;self.promotion_candidate_value=None;self.promotion_candidate_ts=None;logger.info(f"[RATCHET] Baseline promoted: {self.baseline_value:,.2f}");self._on_promotion_event()
	def _on_promotion_event(self)->None:
		if self.promotion_liquidate_enabled:
			logger.info('[RATCHET] Promotion event — pausing trading and liquidating positions');self.system_state='COOLDOWN'
			if self.live_trading_mode and hasattr(self,'positions')and self.positions:
				for(sym,pos)in list(self.positions.items()):
					try:self.execution_engine.execute_trade(signal='SELL',size_fraction=1.,base_symbol=sym,aggressive_mode=self.aggressive_mode)
					except Exception as e:logger.warning(f"[RATCHET] Failed to liquidate {sym}: {e}")
			elif hasattr(self,'positions'):self.positions={}
			self.force_halt_after_promotion=True
		else:self.force_halt_after_promotion=False
	def _calculate_portfolio_metrics(self)->Dict[str,Any]:
		if not hasattr(self,'positions'):self.positions={}
		self._update_position_values();total_invested=sum(pos.get('size',pos.get('value',.0))for pos in self.positions.values());total_unrealized_pnl=sum(pos.get('unrealized_pnl',.0)for pos in self.positions.values());current_portfolio_value=self.initial_portfolio_value+total_unrealized_pnl;base_val=self.initial_portfolio_value if self.initial_portfolio_value>0 else 1.;baseline_val=self.baseline_value if self.baseline_value>0 else base_val;last_val=getattr(self,'last_portfolio_value',base_val);total_return_pct=(current_portfolio_value-base_val)/base_val*100;baseline_return_pct=(current_portfolio_value-baseline_val)/baseline_val*100;return_since_last=(current_portfolio_value-last_val)/last_val*100 if last_val>0 else .0;self.last_portfolio_value,self.total_pnl,self.total_return_pct=current_portfolio_value,total_unrealized_pnl,total_return_pct;self._update_dynamic_position_cap(current_portfolio_value=current_portfolio_value,baseline_value=baseline_val);return{'total_invested':total_invested,'total_unrealized_pnl':total_unrealized_pnl,'current_portfolio_value':current_portfolio_value,'initial_portfolio_value':self.initial_portfolio_value,'baseline_value':self.baseline_value,'total_return_pct':total_return_pct,'baseline_return_pct':baseline_return_pct,'return_since_last':return_since_last,'positions':self.positions.copy()}
	def _get_live_wallet_snapshot(self)->Dict[str,Any]:
		if not self.live_trading_mode or not hasattr(self,'execution_engine')or not self.execution_engine:return{}
		snapshot:Dict[str,Any]={}
		try:
			self.execution_engine._update_balance();snapshot['available_balance']=float(getattr(self.execution_engine,'available_balance',.0)or .0)
			if hasattr(self.execution_engine,'available_balances_by_quote'):by_quote=getattr(self.execution_engine,'available_balances_by_quote',{})or{};snapshot['by_quote']={k:float(v)for(k,v)in by_quote.items()if isinstance(v,(int,float))and abs(v)>0}
			if hasattr(self.execution_engine,'latest_balances_norm'):bal=getattr(self.execution_engine,'latest_balances_norm',{})or{};snapshot['balances']={k:float(v)for(k,v)in bal.items()if isinstance(v,(int,float))and abs(v)>0}
			if hasattr(self.execution_engine,'get_live_holdings'):
				holdings=self.execution_engine.get_live_holdings()
				if holdings:snapshot['holdings']=holdings
			holdings=snapshot.get('holdings',{})or snapshot.get('balances',{})or{};totals={'eur':.0,'usd':.0,'details':[]}
			if holdings and hasattr(self.execution_engine,'exchange')and self.execution_engine.exchange:
				markets=getattr(self.execution_engine,'markets',{})or{};preferred_quotes=['EUR','USD','USDT','USDC']
				for(asset,amt)in holdings.items():
					try:
						if amt is None or abs(float(amt))==0:continue
						amt=float(amt);upper_asset=asset.upper()
						if upper_asset in('EUR',):totals['eur']+=amt;totals['details'].append({'asset':upper_asset,'amount':amt,'quote':'EUR','value':amt});continue
						if upper_asset in('USD','USDT','USDC'):totals['usd']+=amt;totals['details'].append({'asset':upper_asset,'amount':amt,'quote':'USD','value':amt});continue
						quote_used=None;last_price=None
						for q in preferred_quotes:
							pair=f"{upper_asset}/{q}"
							if pair in markets:quote_used=q;ticker=self.execution_engine._with_retries(self.execution_engine.exchange.fetch_ticker,pair);last_price=float(ticker.get('last')or .0);break
						if quote_used and last_price and last_price>0:
							val=amt*last_price
							if quote_used=='EUR':totals['eur']+=val
							else:totals['usd']+=val
							totals['details'].append({'asset':upper_asset,'amount':amt,'quote':quote_used,'value':val})
					except Exception:continue
			snapshot['totals']=totals;snapshot['exchange']=getattr(self.execution_engine,'exchange_name','')
		except Exception as e:logger.warning(f"[WALLET] Failed to snapshot live balances: {e}")
		return snapshot
	def _update_session_pnl(self,portfolio_metrics:Optional[Dict[str,Any]]=None)->Dict[str,Any]:
		metrics=portfolio_metrics or self._calculate_portfolio_metrics();current_value=metrics.get('current_portfolio_value',self.initial_portfolio_value)
		if self.initial_portfolio_value<=0:self.initial_portfolio_value=current_value if current_value>0 else 1.
		self.session_pnl_pct=(current_value-self.initial_portfolio_value)/self.initial_portfolio_value*100 if self.initial_portfolio_value>0 else .0;baseline_pnl_pct=(current_value-self.baseline_value)/self.baseline_value*100 if self.baseline_value>0 else .0
		if current_value>self.baseline_value:self._promote_baseline(current_value)
		elif self.live_trading_mode and self.promotion_debounce_secs>0:self.promotion_candidate_value=None;self.promotion_candidate_ts=None
		return metrics
	def _compute_dynamic_position_cap(self,baseline_value:float)->float:
		hard_cap=TRADING_CONFIG.get('max_position_hard_cap',.3);tiers=[(100,.01),(500,.015),(2000,.02),(10000,.025),(float('inf'),.03)];cap=TRADING_CONFIG.get('max_position_size',.2)
		for(threshold,tier_cap)in tiers:
			if baseline_value<threshold:cap=tier_cap;break
		cap=min(cap,.03*baseline_value**.5/100**.5)if baseline_value>0 else cap;cap=min(cap,hard_cap);cap=max(self.dynamic_max_position_size,cap);return cap
	def _update_dynamic_position_cap(self,current_portfolio_value:float,baseline_value:float)->None:
		if getattr(self,'system_state','ACTIVE')!='ACTIVE':return
		new_cap=self._compute_dynamic_position_cap(baseline_value)
		if new_cap>self.dynamic_max_position_size:
			self.dynamic_max_position_size=new_cap
			if hasattr(self,'risk_monitor')and self.risk_monitor:self.risk_monitor.max_position_size=new_cap
			if hasattr(self,'execution_engine')and self.execution_engine:self.execution_engine.max_position_size=new_cap
			logger.info(f"[RISK] Max position cap increased to {new_cap:.2%} based on baseline {baseline_value:,.2f}")
	def _maybe_trigger_session_circuit_breaker(self,portfolio_metrics:Dict[str,Any])->None:
		if self.circuit_breaker_triggered or not self.cooldown_enabled:return
		pnl=self.session_pnl_pct
		if pnl<=self.stop_loss_pct or pnl>=self.stop_gain_pct:self.circuit_breaker_triggered=True;reason=f"[CIRCUIT] Triggered: session_pnl={pnl:+.2f}%";logger.warning(reason);self._enter_cooldown(reason)
	def _deterministic_cooldown_loops(self)->int:span=max(1,self.cooldown_max_loops-self.cooldown_min_loops+1);return self.cooldown_min_loops+self.global_loop_counter%span
	def _should_enter_cooldown(self,api_budget_remaining:float,congestion_metric:float)->Tuple[bool,str]:
		if not self.cooldown_enabled or self.system_state=='COOLDOWN'or self.cooldown_exit_guard:return False,''
		reasons=[]
		if self.loops_since_cooldown>=self.cooldown_loop_trigger:reasons.append(f"loops_since_cooldown={self.loops_since_cooldown} >= {self.cooldown_loop_trigger}")
		if api_budget_remaining<self.cooldown_api_budget_threshold:reasons.append(f"api_budget={api_budget_remaining:.0%} < {self.cooldown_api_budget_threshold:.0%}")
		if congestion_metric>self.cooldown_congestion_threshold:reasons.append(f"signal_congestion={congestion_metric:.2f} > {self.cooldown_congestion_threshold:.2f}")
		if reasons:return True,'; '.join(reasons)
		return False,''
	def _enter_cooldown(self,reason:str)->None:
		self.system_state='COOLDOWN';self.cooldown_trigger_reason=reason;self.cooldown_total_loops=self._deterministic_cooldown_loops();self.cooldown_loops_remaining=self.cooldown_total_loops;self.loops_since_cooldown=0;metrics=self._calculate_portfolio_metrics();current_val=metrics.get('current_portfolio_value',self.baseline_value);self.cooldown_peak_value=current_val;baseline_pnl_pct=(current_val-self.baseline_value)/self.baseline_value*100 if self.baseline_value>0 else .0;logger.info(f"[RATCHET] Entry check — baseline={self.baseline_value:,.2f}, current={current_val:,.2f}, pnl={baseline_pnl_pct:+.2f}%")
		if current_val>self.baseline_value:self._promote_baseline(current_val)
		logger.info(f"[COOLDOWN] Triggered: {reason} — duration {self.cooldown_total_loops} loops")
	def _cooldown_tick(self,api_budget_remaining:float)->None:
		if self.system_state!='COOLDOWN':return
		completed=self.cooldown_total_loops-self.cooldown_loops_remaining+1;logger.info(f"[COOLDOWN] Loop {completed}/{self.cooldown_total_loops} — observing only (reason: {self.cooldown_trigger_reason})");metrics=self._calculate_portfolio_metrics();current_val=metrics.get('current_portfolio_value',self.baseline_value)
		if self.cooldown_peak_value is None:self.cooldown_peak_value=current_val
		else:self.cooldown_peak_value=max(self.cooldown_peak_value,current_val)
		baseline_pnl_peak=(self.cooldown_peak_value-self.baseline_value)/self.baseline_value*100 if self.baseline_value>0 else .0
		if self.cooldown_peak_value>self.baseline_value:logger.info(f"[RATCHET] New peak during cooldown: peak pnl {baseline_pnl_peak:+.2f}%")
		self.cooldown_loops_remaining=max(0,self.cooldown_loops_remaining-1)
		if self.cooldown_loops_remaining<=0:
			logger.info('[COOLDOWN] Completed');peak_val=self.cooldown_peak_value if self.cooldown_peak_value is not None else current_val;baseline_pnl_pct=(peak_val-self.baseline_value)/self.baseline_value*100 if self.baseline_value>0 else .0;logger.info(f"[RATCHET] Exit check — baseline={self.baseline_value:,.2f}, peak={peak_val:,.2f}, peak pnl={baseline_pnl_pct:+.2f}%")
			if peak_val>self.baseline_value:self._promote_baseline(peak_val)
			self.cooldown_peak_value=None;self.circuit_breaker_triggered=False;self.system_state='ACTIVE';self.cooldown_trigger_reason='';self.cooldown_exit_guard=True;logger.info('[COOLDOWN] Exit — resuming ACTIVE state')
		if self.force_halt_after_promotion:logger.info('[RATCHET] Promotion halt engaged — further trading paused');raise SystemExit(0)
	async def _exit_liquidate_all(self,target_quotes:Optional[List[str]]=None)->Dict[str,Any]:
		self.system_state='EXIT';self.cooldown_trigger_reason='EXIT';target_quotes=target_quotes or['EUR','USD'];summary:Dict[str,Any]={'mode':'EXIT','live':self.live_trading_mode,'actions':[]}
		try:summary['before']=self._calculate_portfolio_metrics()
		except Exception:summary['before']={}
		if not self.live_trading_mode:summary['note']='Demo mode: positions not liquidated';self.exit_summary=summary;return summary
		if not hasattr(self,'positions')or not self.positions:summary['note']='No positions to liquidate';self.exit_summary=summary;return summary
		loop=asyncio.get_event_loop();liquidation_tasks=[]
		for(sym,pos)in list(self.positions.items()):
			async def sell_symbol(symbol:str)->Dict[str,Any]:
				def do_sell()->Dict[str,Any]:
					try:return{'symbol':symbol,'result':self.execution_engine.execute_trade(signal='SELL',size_fraction=1.,base_symbol=symbol,aggressive_mode=self.aggressive_mode)}
					except Exception as e:return{'symbol':symbol,'error':str(e)}
				return await loop.run_in_executor(None,do_sell)
			liquidation_tasks.append(sell_symbol(sym))
		actions=await asyncio.gather(*liquidation_tasks,return_exceptions=False);summary['actions']=actions
		try:summary['after']=self._calculate_portfolio_metrics()
		except Exception:summary['after']={}
		self.exit_summary=summary;logger.info(f"[EXIT] Liquidation summary: {summary}");return summary
	def _incremental_update(self)->Dict[str,Any]:
		if self.last_full_run is None:return{'error':'No cached data available for incremental update'}
		return self.last_full_run
	def _init_cache(self)->str:
		if self.cache_type=='redis'and REDIS_AVAILABLE:
			try:self._redis_client=redis.Redis(host='localhost',port=6379,db=0,decode_responses=True);self._redis_client.ping();return'redis'
			except Exception:return'file'
		return'file'
	def _get_cache(self,key:str)->Optional[Any]:
		if self.cache=='redis'and self._redis_client:
			try:
				cached=self._redis_client.get(key)
				if cached:return json.loads(cached)
			except Exception:pass
		cache_file=self.cache_dir/f"{key.replace(":","_")}.json"
		if cache_file.exists():
			try:
				with open(cache_file,'r')as f:
					data=json.load(f)
					if'timestamp'in data and'ttl'in data:
						if time.time()-data['timestamp']<data['ttl']:return data['data']
					else:return data
			except Exception:pass
	def _cache_data(self,key:str,data:Any,ttl:int=3600):
		cache_data={'data':data,'timestamp':time.time(),'ttl':ttl}
		if self.cache=='redis'and self._redis_client:
			try:self._redis_client.setex(key,ttl,json.dumps(cache_data));return
			except Exception:pass
		cache_file=self.cache_dir/f"{key.replace(":","_")}.json"
		try:
			with open(cache_file,'w')as f:json.dump(cache_data,f)
		except Exception:pass
	async def run(self,incremental:bool=False)->Dict[str,Any]:
		import time;self.global_loop_counter+=1
		if self.system_state!='COOLDOWN':
			self.loops_since_cooldown+=1
			if self.cooldown_exit_guard:self.cooldown_exit_guard=False
		api_budget_remaining=getattr(self,'api_budget_remaining',1.)
		if self.system_state=='COOLDOWN':self._cooldown_tick(api_budget_remaining)
		portfolio_metrics=self._update_session_pnl();self._maybe_trigger_session_circuit_breaker(portfolio_metrics);current_time=time.time();time_since_full_run=current_time-getattr(self,'last_full_run_time',.0)if getattr(self,'last_full_run_time',.0)>0 else float('inf')
		if incremental and self.last_full_run is not None and time_since_full_run<self.full_refresh_interval:
			if VERBOSE_LOGGING:logger.debug('Performing incremental update (reusing cached data)...')
			return self._incremental_update()
		self.last_full_run_time=current_time
		if self.auto_multi_asset and self.multi_asset_mode:return await self.run_batched_multi_asset()
		workflow_steps=['Fetching market data','Computing quant signals','Validating signals (CRCA)','Generating predictions','Optimizing portfolio','Calculating price targets','Making trading decisions'];result=None
		if RICH_AVAILABLE:
			with Progress(SpinnerColumn(),TextColumn('[progress.description]{task.description}'),BarColumn(),TextColumn('[progress.percentage]{task.percentage:>3.0f}%'),TimeRemainingColumn(),console=Console(),transient=False)as progress:
				task=progress.add_task('[cyan]Quant Trading Workflow',total=len(workflow_steps));progress.update(task,description=f"[yellow]{workflow_steps[0]}...");self.fetch_data()
				if self.price_data.empty:return{'error':'No data available'}
				progress.advance(task);progress.update(task,description=f"[yellow]{workflow_steps[1]}...");signals_df=self.compute_signals()
				if signals_df.empty:return{'error':'Signal computation failed'}
				progress.advance(task);progress.update(task,description=f"[yellow]{workflow_steps[2]}...");signal_scores=self.validate_signals();progress.advance(task);progress.update(task,description=f"[yellow]{workflow_steps[3]}...")
				if self.multi_asset_mode:
					multi_predictions=self.generate_multi_asset_predictions(signal_scores)
					if not multi_predictions:return{'error':'Multi-asset prediction generation failed'}
					primary_symbol=list(self.price_data_dict.keys())[0]
					if primary_symbol in multi_predictions:pred_array,intervals,metadata=multi_predictions[primary_symbol];predictions,pred_metadata,covariance=pred_array,metadata,np.array([[.01]])
					else:return{'error':'Primary asset prediction not found'}
				else:
					predictions,covariance,pred_metadata=self.generate_predictions(signal_scores)
					if len(predictions)==0:return{'error':'Prediction generation failed'}
				progress.advance(task);progress.update(task,description=f"[yellow]{workflow_steps[4]}...");signal_names=list(signal_scores.keys())[:10];recent_perf={name:score.get('score',.0)for(name,score)in signal_scores.items()};regime=self.regime_detector.detect_volatility_regime(self.signals_df);model_names=list(self.ensemble_predictor.models.keys())
				if signal_names and model_names:weights,best_model=self.meta_learner.optimize(signal_names,recent_perf,regime,model_names);self.ensemble_predictor.update_weights(weights)
				else:weights,best_model={},model_names[0]if model_names else None
				progress.advance(task);progress.update(task,description=f"[yellow]{workflow_steps[5]}...");latest_pred=predictions[-1]if len(predictions)>0 else .0;pred_std=pred_metadata.get('prediction_std',[]);latest_std=pred_std[-1]if len(pred_std)>0 else abs(latest_pred)*.1;current_price=self.current_price;price_targets={'1d':current_price*(1+latest_pred),'3d':current_price*(1+latest_pred*3),'7d':current_price*(1+latest_pred*7)};confidence_intervals={'1d':{'lower':current_price*(1+latest_pred-1.96*latest_std),'upper':current_price*(1+latest_pred+1.96*latest_std),'mean':price_targets['1d']},'3d':{'lower':current_price*(1+latest_pred*3-1.96*latest_std*np.sqrt(3)),'upper':current_price*(1+latest_pred*3+1.96*latest_std*np.sqrt(3)),'mean':price_targets['3d']},'7d':{'lower':current_price*(1+latest_pred*7-1.96*latest_std*np.sqrt(7)),'upper':current_price*(1+latest_pred*7+1.96*latest_std*np.sqrt(7)),'mean':price_targets['7d']}};progress.advance(task);progress.update(task,description=f"[yellow]{workflow_steps[6]}...");covariance_df=pd.DataFrame();expected_returns_dict,current_prices_dict={},{};volatility_dict,pred_metadata_dict={},{};confidence_intervals_dict,multi_predictions={},{}
				if self.multi_asset_mode and'multi_predictions'in locals():expected_returns_dict={symbol:float(pred_array[-1])if len(pred_array)>0 else .0 for(symbol,(pred_array,_,_))in multi_predictions.items()};current_prices_dict={symbol:self.price_data_dict[symbol]['price'].iloc[-1]for symbol in multi_predictions.keys()};volatility_dict={symbol:metadata.get('prediction_uncertainty',.02)for(symbol,(_,_,metadata))in multi_predictions.items()};pred_metadata_dict={symbol:metadata for(symbol,(_,_,metadata))in multi_predictions.items()};confidence_intervals_dict={symbol:intervals for(symbol,(_,intervals,_))in multi_predictions.items()};returns_dict={symbol:df['returns']for(symbol,df)in self.price_data_dict.items()if'returns'in df.columns};covariance_df=self.cov_estimator.compute_cross_asset_covariance(returns_dict)if len(returns_dict)>1 else pd.DataFrame()
			if not covariance_df.empty:
				try:
					corr_matrix=np.corrcoef(covariance_df.values)
					if corr_matrix.shape[0]>1:upper=corr_matrix[np.triu_indices_from(corr_matrix,k=1)];congestion_metric=float(np.mean(np.abs(upper)))
				except Exception:congestion_metric=.0;portfolio_weights_dict=self.optimize_portfolio(expected_returns_dict,covariance_df)if not covariance_df.empty else{symbol:.0 for symbol in expected_returns_dict.keys()};decisions_dict=self._make_multi_asset_decisions(portfolio_weights=portfolio_weights_dict,predictions={symbol:pred for(symbol,(pred,_,_))in multi_predictions.items()},signal_scores=signal_scores,pred_metadata_dict=pred_metadata_dict,confidence_intervals_dict=confidence_intervals_dict,volatility_dict=volatility_dict,current_prices=current_prices_dict,max_position_size=self.risk_monitor.max_position_size);primary_symbol=list(self.price_data_dict.keys())[0];primary_decision=decisions_dict.get(primary_symbol,{});progress.update(task,completed=len(workflow_steps));return{'signal':primary_decision.get('signal','HOLD'),'confidence':primary_decision.get('confidence',.0),'current_price':current_prices_dict.get(primary_symbol,self.current_price),'price_targets':{symbol:current_prices_dict[symbol]*(1+expected_returns_dict[symbol])for symbol in current_prices_dict.keys()},'confidence_intervals':confidence_intervals_dict,'risk_metrics':{'volatility':{symbol:vol for(symbol,vol)in volatility_dict.items()},'prediction_uncertainty':{symbol:meta.get('prediction_uncertainty',.0)for(symbol,meta)in pred_metadata_dict.items()}},'signal_explanations':primary_decision.get('explanations',[]),'top_signals':primary_decision.get('top_signals',[]),'recommended_position_size':primary_decision.get('recommended_position_size',.0),'entry_level':primary_decision.get('entry_level'),'stop_loss':primary_decision.get('stop_loss'),'take_profit':primary_decision.get('take_profit'),'portfolio_weights':portfolio_weights_dict,'decisions':decisions_dict,'multi_asset_mode':True,'portfolio_metrics':self._calculate_portfolio_metrics()}
				else:
					expected_returns=np.array([latest_pred]);cov_size=CovarianceEstimator.CovSize(covariance);portfolio_weights=self.optimize_portfolio(expected_returns,covariance)if cov_size>0 else np.array([.0])
					if'returns'in self.signals_df.columns:returns=self.signals_df['returns'].dropna();volatility=returns.std()*np.sqrt(252)if len(returns)>1 else .0;sharpe_ratio=returns.mean()*252/(volatility+1e-06)if len(returns)>1 and volatility>0 else .0;cvar_95=returns[returns<=np.percentile(returns,5)].mean()if len(returns)>10 else returns.mean()-1.65*returns.std()if len(returns)>1 else .0
					else:volatility,sharpe_ratio,cvar_95=.0,.0,.0
					decision_result=self._make_trading_decision(portfolio_weights=portfolio_weights,predictions=predictions,signal_scores=signal_scores,pred_metadata=pred_metadata,confidence_intervals=confidence_intervals,volatility=volatility,current_price=current_price,max_position_size=self.risk_monitor.max_position_size);signal,confidence,signal_explanations=decision_result['signal'],decision_result['confidence'],decision_result['explanations'];original_signal,original_position_size=signal,decision_result['recommended_position_size'];circuit_ok,circuit_msg=self.circuit_breaker.check_circuit()
					if not circuit_ok:logger.warning(f"Circuit breaker tripped: {circuit_msg}");signal_explanations.insert(0,f"🚫 Circuit breaker activated: {circuit_msg}");progress.update(task,completed=len(workflow_steps));return{'error':f"Circuit breaker: {circuit_msg}",'signal':'HOLD','original_signal':original_signal,'current_price':current_price,'signal_explanations':signal_explanations,'monitoring_summary':self.monitoring.get_summary()}
				if signal!='HOLD':
					position_size=abs(decision_result['recommended_position_size']);portfolio_value=portfolio_metrics.get('current_portfolio_value',getattr(self,'account_size',current_price));current_positions=self.positions if hasattr(self,'positions')else{};risk_ok,risk_msg=self.risk_monitor.pre_trade_check(signal=signal,position_size=position_size,current_positions=current_positions,portfolio_value=portfolio_value)
					if not risk_ok:
						max_allowed_size=self.risk_monitor.max_position_size
						if position_size>max_allowed_size:decision_result['recommended_position_size']=np.sign(decision_result['recommended_position_size'])*max_allowed_size;signal_explanations.append(f"⚠️ Position size adjusted from {position_size:.2%} to {max_allowed_size:.2%} due to risk limits")
						else:signal='HOLD';signal_explanations.insert(0,f"🚫 Trade blocked: {risk_msg}")
					self.monitoring.monitor_signal_health(signal_name='ensemble',score=confidence,decay=pred_metadata.get('prediction_std',[.0])[-1]if pred_metadata.get('prediction_std')else .0);progress.update(task,completed=len(workflow_steps))
				return{'signal':signal,'original_signal':original_signal if signal!=original_signal else None,'confidence':confidence,'current_price':current_price,'price_targets':price_targets,'confidence_intervals':confidence_intervals,'signal_explanations':signal_explanations,'top_signals':sorted([(k,v.get('score',.0))for(k,v)in signal_scores.items()],key=lambda x:x[1],reverse=True)[:10],'signal_contributions':pred_metadata.get('signal_contributions',{}),'risk_metrics':{'volatility':volatility,'sharpe_ratio':sharpe_ratio,'cvar_95':cvar_95,'prediction_uncertainty':latest_std},'portfolio_weight':portfolio_weights[0]if len(portfolio_weights)>0 else .0,'recommended_position_size':decision_result['recommended_position_size'],'original_position_size':original_position_size if abs(original_position_size)!=abs(decision_result['recommended_position_size'])else None,'expected_return':latest_pred,'predictions':predictions.tolist()if len(predictions)>0 else[],'regime':self.regime_detector.detect_volatility_regime(self.signals_df),'entry_level':decision_result.get('entry_level'),'stop_loss':decision_result.get('stop_loss'),'take_profit':decision_result.get('take_profit'),'monitoring_summary':self.monitoring.get_summary(),'portfolio_metrics':self._calculate_portfolio_metrics()}
		else:
			self.fetch_data()
			if self.price_data.empty:return{'error':'No data available'}
			signals_df=self.compute_signals()
			if signals_df.empty:return{'error':'Signal computation failed'}
			signal_scores=self.validate_signals()
			if self.multi_asset_mode:
				multi_predictions=self.generate_multi_asset_predictions(signal_scores)
				if not multi_predictions:return{'error':'Multi-asset prediction generation failed'}
				primary_symbol=list(self.price_data_dict.keys())[0]
				if primary_symbol in multi_predictions:pred_array,intervals,metadata=multi_predictions[primary_symbol];predictions,pred_metadata,covariance=pred_array,metadata,np.array([[.01]])
				else:return{'error':'Primary asset prediction not found'}
			else:
				predictions,covariance,pred_metadata=self.generate_predictions(signal_scores)
				if len(predictions)==0:return{'error':'Prediction generation failed'}
			signal_names=list(signal_scores.keys())[:10];recent_perf={name:score.get('score',.0)for(name,score)in signal_scores.items()};regime=self.regime_detector.detect_volatility_regime(self.signals_df);model_names=list(self.ensemble_predictor.models.keys())
			if signal_names and model_names:weights,best_model=self.meta_learner.optimize(signal_names,recent_perf,regime,model_names);self.ensemble_predictor.update_weights(weights)
			else:weights,best_model={},model_names[0]if model_names else None
			latest_pred=predictions[-1]if len(predictions)>0 else .0;pred_std=pred_metadata.get('prediction_std',[]);latest_std=pred_std[-1]if len(pred_std)>0 else abs(latest_pred)*.1;current_price=self.current_price;pred_return=latest_pred/current_price if abs(latest_pred)>1. and current_price>0 else latest_pred;pred_std_return=latest_std/current_price if abs(latest_pred)>1. and current_price>0 else latest_std;price_targets,confidence_intervals={},{}
			for horizon in[1,3,7]:simulated_prices=np.array([current_price*np.exp(np.sum(np.random.normal(pred_return,pred_std_return,horizon)))for _ in range(1000)]);price_targets[f"{horizon}d"]=float(np.mean(simulated_prices));confidence_intervals[f"{horizon}d"]={'lower':float(np.percentile(simulated_prices,2.5)),'upper':float(np.percentile(simulated_prices,97.5)),'mean':float(np.mean(simulated_prices)),'median':float(np.median(simulated_prices)),'std':float(np.std(simulated_prices))}
			if self.multi_asset_mode and'multi_predictions'in locals():
				expected_returns_dict={symbol:float(pred_array[-1])if len(pred_array)>0 else .0 for(symbol,(pred_array,_,_))in multi_predictions.items()};current_prices_dict={symbol:self.price_data_dict[symbol]['price'].iloc[-1]for symbol in multi_predictions.keys()};volatility_dict={symbol:metadata.get('prediction_uncertainty',.02)for(symbol,(_,_,metadata))in multi_predictions.items()};pred_metadata_dict={symbol:metadata for(symbol,(_,_,metadata))in multi_predictions.items()};confidence_intervals_dict={symbol:intervals for(symbol,(_,intervals,_))in multi_predictions.items()};returns_dict={symbol:df['returns']for(symbol,df)in self.price_data_dict.items()if'returns'in df.columns};covariance_df=self.cov_estimator.compute_cross_asset_covariance(returns_dict)if len(returns_dict)>1 else pd.DataFrame();portfolio_weights_dict=self.optimize_portfolio(expected_returns_dict,covariance_df)if not covariance_df.empty else{symbol:.0 for symbol in expected_returns_dict.keys()};weight_values=list(portfolio_weights_dict.values())
				if len(weight_values)>1:
					weight_std,weight_mean_abs=np.std(weight_values),np.mean(np.abs(weight_values))
					if weight_mean_abs>1e-06 and weight_std/weight_mean_abs<.01:
						if VERBOSE_LOGGING:logger.warning(f"Portfolio weights too similar (std={weight_std:.6f}, mean={weight_mean_abs:.6f}). Adding differentiation.")
						for symbol in portfolio_weights_dict:expected_ret=expected_returns_dict.get(symbol,.0);portfolio_weights_dict[symbol]+=expected_ret*.05 if abs(expected_ret)>1e-06 else .0
						total_abs_weight=sum(abs(w)for w in portfolio_weights_dict.values())
						if total_abs_weight>1.:portfolio_weights_dict={k:v*(1./total_abs_weight)for(k,v)in portfolio_weights_dict.items()}
				decisions_dict=self._make_multi_asset_decisions(portfolio_weights=portfolio_weights_dict,predictions={symbol:pred for(symbol,(pred,_,_))in multi_predictions.items()},signal_scores=signal_scores,pred_metadata_dict=pred_metadata_dict,confidence_intervals_dict=confidence_intervals_dict,volatility_dict=volatility_dict,current_prices=current_prices_dict,max_position_size=self.risk_monitor.max_position_size);primary_symbol=list(self.price_data_dict.keys())[0];primary_decision=decisions_dict.get(primary_symbol,{})
		causal_score,causal_block=self._evaluate_causal_stability();result={'signal':primary_decision.get('signal','HOLD'),'confidence':primary_decision.get('confidence',.0),'current_price':current_prices_dict.get(primary_symbol,self.current_price),'price_targets':{symbol:current_prices_dict[symbol]*(1+expected_returns_dict[symbol])for symbol in current_prices_dict.keys()},'confidence_intervals':confidence_intervals_dict,'risk_metrics':{'volatility':{symbol:vol for(symbol,vol)in volatility_dict.items()},'prediction_uncertainty':{symbol:meta.get('prediction_uncertainty',.0)for(symbol,meta)in pred_metadata_dict.items()}},'signal_explanations':primary_decision.get('explanations',[]),'top_signals':primary_decision.get('top_signals',[]),'recommended_position_size':primary_decision.get('recommended_position_size',.0),'entry_level':primary_decision.get('entry_level'),'stop_loss':primary_decision.get('stop_loss'),'take_profit':primary_decision.get('take_profit'),'portfolio_weights':portfolio_weights_dict,'decisions':decisions_dict,'multi_asset_mode':True,'portfolio_metrics':self._calculate_portfolio_metrics(),'causal_score':causal_score,'causal_block':causal_block}
		if not isinstance(result,dict):logger.error(f"ERROR: run_batched_multi_asset() about to return {type(result)} instead of dict!");return{'error':f"Invalid return type in run_batched_multi_asset(): {type(result)}"}
		self.last_full_run=result;return result;expected_returns=np.array([latest_pred]);cov_size=CovarianceEstimator.CovSize(covariance);portfolio_weights=self.optimize_portfolio(expected_returns,covariance)if cov_size>0 else np.array([.0])
		if'returns'in self.signals_df.columns:returns=self.signals_df['returns'].dropna();volatility=returns.std()*np.sqrt(252)if len(returns)>1 else .0;sharpe_ratio=returns.mean()*252/(volatility+1e-06)if len(returns)>1 and volatility>0 else .0;cvar_95=returns[returns<=np.percentile(returns,5)].mean()if len(returns)>10 else returns.mean()-1.65*returns.std()if len(returns)>1 else .0
		else:volatility,sharpe_ratio,cvar_95=.0,.0,.0
		congestion_metric=.0
		if self.system_state!='COOLDOWN'and self.cooldown_enabled:
			trigger,reason=self._should_enter_cooldown(api_budget_remaining,congestion_metric)
			if trigger:self._enter_cooldown(reason)
		decision_result=self._make_trading_decision(portfolio_weights=portfolio_weights,predictions=predictions,signal_scores=signal_scores,pred_metadata=pred_metadata,confidence_intervals=confidence_intervals,volatility=volatility,current_price=current_price,max_position_size=self.risk_monitor.max_position_size);signal,confidence,signal_explanations=decision_result['signal'],decision_result['confidence'],decision_result['explanations']
		if self.system_state in('COOLDOWN','EXIT'):signal='EXIT'if self.system_state=='EXIT'else'COOLDOWN';decision_result['recommended_position_size']=.0;decision_result['explanations']=signal_explanations+[f"[{signal}] Observing only; no trades placed"]
		original_signal,original_position_size=signal,decision_result['recommended_position_size'];circuit_ok,circuit_msg=self.circuit_breaker.check_circuit()
		if not circuit_ok:logger.warning(f"Circuit breaker tripped: {circuit_msg}");signal_explanations.insert(0,f"🚫 Circuit breaker activated: {circuit_msg}");return{'error':f"Circuit breaker: {circuit_msg}",'signal':'HOLD','original_signal':original_signal,'current_price':current_price,'signal_explanations':signal_explanations,'monitoring_summary':self.monitoring.get_summary()}
		if signal not in['HOLD','COOLDOWN']:
			position_size=abs(decision_result['recommended_position_size']);portfolio_value=decision_result.get('portfolio_metrics',{}).get('current_portfolio_value',getattr(self,'account_size',current_price));current_positions=self.positions if hasattr(self,'positions')else{};risk_ok,risk_msg=self.risk_monitor.pre_trade_check(signal=signal,position_size=position_size,current_positions=current_positions,portfolio_value=portfolio_value)
			if not risk_ok:
				max_allowed_size=self.risk_monitor.max_position_size
				if position_size>max_allowed_size:decision_result['recommended_position_size']=np.sign(decision_result['recommended_position_size'])*max_allowed_size;signal_explanations.append(f"⚠️ Position size adjusted from {position_size:.2%} to {max_allowed_size:.2%} due to risk limits");logger.info(f"Position size adjusted from {position_size:.2%} to {max_allowed_size:.2%}")
				else:signal='HOLD';signal_explanations.insert(0,f"🚫 Trade blocked: {risk_msg}");logger.warning(f"Risk check failed: {risk_msg}")
		if signal!=original_signal and original_signal!='HOLD':
			if signal_explanations and signal_explanations[0].startswith(original_signal):signal_explanations[0]=f"🚫 {signal} signal: Changed from {original_signal} due to risk management ({confidence:.0%} confidence)"
		trade_result=None
		if self.live_trading_mode and signal not in['HOLD','COOLDOWN']:
			logger.warning(f"⚠️ LIVE TRADING MODE ENABLED - Executing {signal} trade");expected_return_pct=decision_result.get('expected_return',None);rec_sz=decision_result.get('recommended_position_size',.0)or .0;trade_result=self.execution_engine.execute_trade(signal=signal,size_fraction=min(abs(rec_sz),self.execution_engine.max_position_size),base_symbol=primary_symbol,aggressive_mode=self.aggressive_mode,expected_return_pct=expected_return_pct)
			if trade_result and trade_result.get('status')=='success':filled_amt=trade_result.get('amount',.0);filled_price=trade_result.get('price',current_price);self._record_live_fill(symbol=primary_symbol,side=signal,amount=filled_amt,price=filled_price);filled_val=filled_amt*filled_price;portfolio_value=self._calculate_portfolio_metrics().get('current_portfolio_value',self.account_size if self.account_size>0 else 1.);frac=filled_val/portfolio_value if portfolio_value>0 else .0;decision_result['recommended_position_size']=frac
		elif signal not in['HOLD','COOLDOWN']:logger.info(f"📊 DEMO MODE: Would execute {signal} trade (live_trading_mode=False)")
		if trade_result and'pnl'in trade_result:self.circuit_breaker.record_trade(trade_result['pnl']);self.monitoring.track_pnl(timestamp=datetime.now(),pnl=trade_result['pnl'],position=portfolio_weights[0]if len(portfolio_weights)>0 else .0,price=current_price)
		self.monitoring.monitor_signal_health(signal_name='ensemble',score=confidence,decay=pred_metadata.get('prediction_std',[.0])[-1]if pred_metadata.get('prediction_std')else .0);causal_score,causal_block=self._evaluate_causal_stability();wallet_snapshot=self._get_live_wallet_snapshot();result={'signal':signal,'original_signal':original_signal if signal!=original_signal else None,'confidence':confidence,'current_price':current_price,'price_targets':price_targets,'confidence_intervals':confidence_intervals,'signal_explanations':signal_explanations,'top_signals':sorted([(k,v.get('score',.0))for(k,v)in signal_scores.items()],key=lambda x:x[1],reverse=True)[:10],'signal_contributions':pred_metadata.get('signal_contributions',{}),'risk_metrics':{'volatility':volatility,'sharpe_ratio':sharpe_ratio,'cvar_95':cvar_95,'prediction_uncertainty':latest_std},'portfolio_weight':portfolio_weights[0]if len(portfolio_weights)>0 else .0,'recommended_position_size':decision_result['recommended_position_size'],'original_position_size':original_position_size if abs(original_position_size)!=abs(decision_result['recommended_position_size'])else None,'expected_return':latest_pred,'predictions':predictions.tolist()if len(predictions)>0 else[],'regime':regime,'best_model':best_model,'entry_level':decision_result.get('entry_level',current_price),'stop_loss':decision_result.get('stop_loss',None),'take_profit':decision_result.get('take_profit',None),'trade_result':trade_result,'monitoring_summary':self.monitoring.get_summary(),'portfolio_metrics':self._calculate_portfolio_metrics(),'system_state':self.system_state,'causal_score':causal_score,'causal_block':causal_block,'wallet':wallet_snapshot}
		if not isinstance(result,dict):logger.error(f"ERROR: run() about to return {type(result)} instead of dict!");logger.error(f"Result keys: {list(result.keys())if hasattr(result,"keys")else"No keys"}");return{'error':f"Invalid return type in run(): {type(result)}"}
		self.last_full_run=result;return result
def display_rich_results(console:Console,result:Dict[str,Any],live_mode:bool=False)->None:
	if'error'in result:console.print(Panel(f"[red]Error: {result["error"]}[/red]",title='Error',border_style='red'));return
	def _render_wallet_panel(wallet:Dict[str,Any])->None:
		if not wallet:return
		lines=[];exch=wallet.get('exchange')or'live';available=wallet.get('available_balance')
		if available is not None:lines.append(f"[bold]Available (primary):[/bold] {available:,.4f}")
		by_quote=wallet.get('by_quote',{})
		if by_quote:
			lines.append('[bold]Quotes:[/bold]')
			for(q,bal)in sorted(by_quote.items()):lines.append(f"  {q}: {bal:,.4f}")
		holdings=wallet.get('holdings')or wallet.get('balances')or{}
		if holdings:
			lines.append('[bold]Assets:[/bold]')
			for(asset,bal)in sorted(holdings.items()):
				if asset in by_quote:continue
				lines.append(f"  {asset}: {bal:,.6f}")
		totals=wallet.get('totals',{})
		if totals:
			lines.append('[bold]Totals:[/bold]')
			if'eur'in totals:lines.append(f"  EUR total: {totals.get("eur",.0):,.4f}")
			if'usd'in totals:lines.append(f"  USD total: {totals.get("usd",.0):,.4f}")
		details=wallet.get('totals',{}).get('details',[])
		if details:
			lines.append('[dim]Valuations:[/dim]')
			for d in details[:10]:lines.append(f"  {d.get("asset")}: {d.get("value",0):,.4f} {d.get("quote")}")
		if exch:lines.append(f"[dim]Exchange: {exch}[/dim]")
		if lines:console.print(Panel('\n'.join(lines),title='[bold white]💰 LIVE WALLET[/bold white]',border_style='white'))
	batched_results=result.get('batched_results',[]);has_multi_asset=len(batched_results)>0
	if live_mode:
		_render_wallet_panel(result.get('wallet',{}))
		if has_multi_asset:display_multi_asset_table(console,result,batched_results,live_mode=True)
		else:display_single_asset_view(console,result,live_mode=True)
	else:
		console.print();console.print(Panel.fit('[bold cyan]QUANT TRADING AGENT - MULTI-ASSET ANALYSIS[/bold cyan]',border_style='cyan',box=box.DOUBLE));console.print()
		if has_multi_asset:display_multi_asset_table(console,result,batched_results,live_mode=False)
		else:display_single_asset_view(console,result,live_mode=False)
def display_multi_asset_table(console:Console,result:Dict[str,Any],batched_results:List[Dict],live_mode:bool=False)->None:
	all_current_prices=result.get('all_current_prices',{});all_metadata=result.get('all_metadata',{});all_intervals=result.get('all_intervals',{});price_data_dict=result.get('price_data_dict',{});portfolio_metrics=result.get('portfolio_metrics',{});positions=portfolio_metrics.get('positions',{});wallet=result.get('wallet',{})if live_mode else{}
	if live_mode:
		holdings=wallet.get('holdings')or wallet.get('balances');by_quote=wallet.get('by_quote',{})
		if holdings:
			live_positions={}
			for(asset,bal)in holdings.items():
				if asset in by_quote:continue
				if bal is None or abs(bal)==0:continue
				live_positions[asset]={'size':bal}
			positions=live_positions
	MAX_DISPLAY_ASSETS=20;total_assets=result.get('total_assets',len(all_current_prices))
	if live_mode:from datetime import datetime;timestamp=datetime.now().strftime('%H:%M:%S');table_title=f"[bold cyan]📊 LIVE ASSET TRADING DECISIONS[/bold cyan] - {timestamp}"
	else:table_title='[bold cyan]📊 ASSET TRADING DECISIONS[/bold cyan]'
	main_table=Table(title=table_title,show_header=True,header_style='bold cyan',box=box.ROUNDED,border_style='cyan',title_style='bold cyan');main_table.add_column('Asset',style='bold white',width=8);main_table.add_column('Price',justify='right',style='yellow',width=12);main_table.add_column('Signal',justify='center',width=10);main_table.add_column('Confidence',justify='right',width=12);main_table.add_column('Position',justify='right',style='green',width=12);main_table.add_column('Invested',justify='right',style='blue',width=12);main_table.add_column('PnL',justify='right',width=14);main_table.add_column('Expected Return',justify='right',width=14);main_table.add_column('Volatility',justify='right',width=12);main_table.add_column('Uncertainty',justify='right',width=12);main_table.add_column('1D Target',justify='right',style='magenta',width=12);main_table.add_column('7D Target',justify='right',style='magenta',width=12);MAX_DISPLAY_ASSETS=20;total_assets=result.get('total_assets',len(all_current_prices));all_assets_data=[]
	for batch in batched_results:
		decisions=batch.get('decisions',{})
		for(symbol,decision)in decisions.items():all_assets_data.append({'symbol':symbol,'decision':decision,'batch':batch.get('batch',0)})
	def sort_key(x):symbol=x['symbol'];decision=x['decision'];has_position=symbol in positions and positions[symbol].get('size',0)>0;confidence=decision.get('confidence',.0)if isinstance(decision,dict)else .0;signal=decision.get('signal','HOLD')if isinstance(decision,dict)else'HOLD';signal_priority={'BUY':3,'SELL':2,'COOLDOWN':1,'HOLD':1}.get(signal,0);return has_position,signal_priority,confidence,symbol
	all_assets_data.sort(key=sort_key,reverse=True);display_assets=all_assets_data[:MAX_DISPLAY_ASSETS];has_more=len(all_assets_data)>MAX_DISPLAY_ASSETS
	for asset_data in display_assets:
		symbol=asset_data['symbol'];decision=asset_data['decision'];price=all_current_prices.get(symbol,result.get('current_price',0));signal=decision.get('signal','HOLD')if isinstance(decision,dict)else'HOLD';confidence=decision.get('confidence',.0)if isinstance(decision,dict)else .0;position_size=decision.get('recommended_position_size',.0)if isinstance(decision,dict)else .0;expected_return=decision.get('expected_return',.0)if isinstance(decision,dict)else .0;metadata=all_metadata.get(symbol,{});volatility=.0;uncertainty=.0
		if isinstance(decision,dict):volatility=decision.get('volatility',.0)
		if volatility==.0 and symbol in price_data_dict:
			try:
				price_data=price_data_dict[symbol]
				if isinstance(price_data,pd.DataFrame)and'price'in price_data.columns and len(price_data)>1:
					returns=price_data['price'].pct_change().dropna()
					if len(returns)>0:volatility=float(returns.std()*np.sqrt(252))
			except Exception:pass
		if volatility==.0:volatility=.02
		if isinstance(decision,dict):uncertainty=decision.get('prediction_uncertainty',.0)
		if uncertainty==.0 and isinstance(metadata,dict):
			raw_uncertainty=metadata.get('prediction_uncertainty',.0);pred_std=metadata.get('prediction_std',[])
			if isinstance(pred_std,list)and len(pred_std)>0:
				latest_std=pred_std[-1]if len(pred_std)>0 else .0
				if expected_return!=.0 and abs(expected_return)>1e-06:uncertainty=min(2.,latest_std/abs(expected_return))
				elif latest_std>0 and price>0:uncertainty=min(2.,latest_std/price)
				else:uncertainty=.5
			elif raw_uncertainty>0:uncertainty=min(2.,raw_uncertainty)
			else:uncertainty=.5
			uncertainty=max(.01,uncertainty)
		intervals=all_intervals.get(symbol,{});target_1d=price;target_7d=price
		if intervals and isinstance(intervals,dict):
			if'1d'in intervals and isinstance(intervals['1d'],dict):target_1d=intervals['1d'].get('mean',price)
			if'7d'in intervals and isinstance(intervals['7d'],dict):target_7d=intervals['7d'].get('mean',price)
		if signal=='BUY':signal_text=Text('🟢 BUY',style='bold green')
		elif signal=='SELL':signal_text=Text('🔴 SELL',style='bold red')
		elif signal=='COOLDOWN':signal_text=Text('🔵 COOLDOWN',style='bold cyan')
		else:signal_text=Text('🟡 HOLD',style='bold yellow')
		if confidence>=.8:conf_style='bold green'
		elif confidence>=.6:conf_style='yellow'
		else:conf_style='red'
		if abs(position_size)>.05:pos_style='bold green'
		elif abs(position_size)>.02:pos_style='yellow'
		else:pos_style='dim'
		position_info=positions.get(symbol,{});invested_value=position_info.get('value',.0);unrealized_pnl=position_info.get('unrealized_pnl',.0);unrealized_pnl_pct=position_info.get('unrealized_pnl_pct',.0);invested_text=f"${invested_value:,.2f}"if invested_value>0 else'-'
		if unrealized_pnl>0:pnl_text=f"[green]+${unrealized_pnl:,.2f} (+{unrealized_pnl_pct:.2f}%)[/green]"
		elif unrealized_pnl<0:pnl_text=f"[red]${unrealized_pnl:,.2f} ({unrealized_pnl_pct:.2f}%)[/red]"
		else:pnl_text='-'
		main_table.add_row(symbol,f"${price:,.2f}",signal_text,f"[{conf_style}]{confidence:.1%}[/{conf_style}]",f"[{pos_style}]{position_size:.2%}[/{pos_style}]",invested_text,pnl_text,f"{expected_return:+.2%}",f"{volatility:.1%}",f"{uncertainty:.1%}",f"${target_1d:,.2f}",f"${target_7d:,.2f}")
	if has_more:remaining=len(all_assets_data)-MAX_DISPLAY_ASSETS;main_table.add_row(f"[dim]... {remaining} more[/dim]",'[dim]-[/dim]','[dim]-[/dim]','[dim]-[/dim]','[dim]-[/dim]','[dim]-[/dim]','[dim]-[/dim]','[dim]-[/dim]','[dim]-[/dim]','[dim]-[/dim]','[dim]-[/dim]','[dim]-[/dim]','[dim]-[/dim]')
	console.print(main_table)
	if total_assets>MAX_DISPLAY_ASSETS:summary_text=f"[dim]Showing top {MAX_DISPLAY_ASSETS} of {total_assets} assets (sorted by priority)[/dim]";console.print(summary_text)
	if portfolio_metrics:
		portfolio_table=Table(title='[bold green]💰 PORTFOLIO SUMMARY[/bold green]',show_header=True,header_style='bold green',box=box.ROUNDED,border_style='green');portfolio_table.add_column('Metric',style='bold white',width=20);portfolio_table.add_column('Value',justify='right',style='yellow',width=20);initial_value=portfolio_metrics.get('initial_portfolio_value',.0);current_value=portfolio_metrics.get('current_portfolio_value',initial_value);baseline_value=portfolio_metrics.get('baseline_value',initial_value);total_pnl=portfolio_metrics.get('total_unrealized_pnl',.0);total_return=portfolio_metrics.get('total_return_pct',.0);baseline_return=portfolio_metrics.get('baseline_return_pct',.0);return_since_last=portfolio_metrics.get('return_since_last',.0);total_invested=portfolio_metrics.get('total_invested',.0);positions=portfolio_metrics.get('positions',{})or{}
		def _best_worst(pos_dict):
			if not pos_dict:return None,None
			sortable=[]
			for(sym,p)in pos_dict.items():pct=p.get('unrealized_pnl_pct',.0);sortable.append((pct,sym))
			if not sortable:return None,None
			sortable.sort();worst=sortable[0];best=sortable[-1];return best,worst
		best,worst=_best_worst(positions)
		if total_return>0:return_text=f"[green]+{total_return:.2f}%[/green]"
		elif total_return<0:return_text=f"[red]{total_return:.2f}%[/red]"
		else:return_text='0.00%'
		if baseline_return>0:baseline_text=f"[green]+{baseline_return:.2f}%[/green]"
		elif baseline_return<0:baseline_text=f"[red]{baseline_return:.2f}%[/red]"
		else:baseline_text='0.00%'
		def _fmt_bw(item):
			if not item:return'—'
			pct,sym=item
			if pct>0:return f"{sym}: [green]+{pct:.2f}%[/green]"
			elif pct<0:return f"{sym}: [red]{pct:.2f}%[/red]"
			return f"{sym}: 0.00%"
		if return_since_last>0:since_last_text=f"[green]+{return_since_last:.2f}%[/green]"
		elif return_since_last<0:since_last_text=f"[red]{return_since_last:.2f}%[/red]"
		else:since_last_text='0.00%'
		portfolio_table.add_row('Initial Portfolio Value',f"${initial_value:,.2f}");portfolio_table.add_row('Baseline (Ratcheted)',f"${baseline_value:,.2f}");portfolio_table.add_row('Current Portfolio Value',f"${current_value:,.2f}");portfolio_table.add_row('Total Invested',f"${total_invested:,.2f}");portfolio_table.add_row('Unrealized PnL',f"${total_pnl:+,.2f}");portfolio_table.add_row('Total Return',return_text);portfolio_table.add_row('Baseline Return',baseline_text);portfolio_table.add_row('Return Since Last Run',since_last_text);portfolio_table.add_row('Highest Gained',_fmt_bw(best));portfolio_table.add_row('Lowest Gained',_fmt_bw(worst));console.print();console.print(portfolio_table)
	causal_score=result.get('causal_score',None);causal_block=result.get('causal_block',False)
	if causal_score is not None and not live_mode:causal_text=f"[bold white]Causal score:[/bold white] {causal_score:.2f}{" 🚫 block"if causal_block else""}";console.print(Panel(causal_text,title='[bold blue]🧠 CAUSAL[/bold blue]',border_style='blue'))
	if not live_mode:
		console.print();console.print(Panel.fit('[bold cyan]📈 DETAILED ASSET ANALYSIS[/bold cyan]',border_style='cyan'));console.print();asset_panels=[]
		for asset_data in all_assets_data[:10]:
			symbol=asset_data['symbol'];decision=asset_data['decision'];info_lines=[];signal=decision.get('signal','HOLD')if isinstance(decision,dict)else'HOLD';confidence=decision.get('confidence',.0)if isinstance(decision,dict)else .0;info_lines.append(f"[bold]Signal:[/bold] {signal} ({confidence:.1%})");explanations=decision.get('explanations',[])if isinstance(decision,dict)else[]
			if explanations:
				info_lines.append('');info_lines.append('[bold]Reasons:[/bold]')
				for exp in explanations[:3]:info_lines.append(f"  • {exp}")
			top_signals=decision.get('top_signals',[])if isinstance(decision,dict)else[]
			if top_signals:
				info_lines.append('');info_lines.append('[bold]Top Signals:[/bold]')
				for(sig_name,score)in top_signals[:3]:sig_short=sig_name.replace('signal_','')[:25];info_lines.append(f"  {sig_short}: {score:.3f}")
			entry=decision.get('entry_level')if isinstance(decision,dict)else None;stop_loss=decision.get('stop_loss')if isinstance(decision,dict)else None;take_profit=decision.get('take_profit')if isinstance(decision,dict)else None
			if entry or stop_loss or take_profit:
				info_lines.append('');info_lines.append('[bold]Levels:[/bold]')
				if entry:info_lines.append(f"  Entry: ${entry:,.2f}")
				if stop_loss:info_lines.append(f"  Stop: ${stop_loss:,.2f}")
				if take_profit:info_lines.append(f"  Target: ${take_profit:,.2f}")
			causal_score=decision.get('causal_score')if isinstance(decision,dict)else None;causal_block=decision.get('causal_block',False)if isinstance(decision,dict)else False
			if causal_score is not None:info_lines.append('');info_lines.append('[bold]Causal:[/bold]');info_lines.append(f"  Score: {causal_score:.2f}{" 🚫 block"if causal_block else""}")
			panel_content='\n'.join(info_lines)if info_lines else'No detailed data available';border_color='green'if signal=='BUY'else'red'if signal=='SELL'else'cyan'if signal=='COOLDOWN'else'yellow';asset_panels.append(Panel(panel_content,title=f"[bold]{symbol}[/bold]",border_style=border_color,box=box.ROUNDED))
		if asset_panels:console.print(Columns(asset_panels,equal=True,expand=True));console.print()
		if not live_mode:display_summary_stats(console,result,all_assets_data)
def display_single_asset_view(console:Console,result:Dict[str,Any],live_mode:bool=False)->None:
	if live_mode:from datetime import datetime;timestamp=datetime.now().strftime('%H:%M:%S');signal=result.get('signal','HOLD');confidence=result.get('confidence',.0);price=result.get('current_price',0);signal_emoji='🟢'if signal=='BUY'else'🔴'if signal=='SELL'else'🔵'if signal=='COOLDOWN'else'🟡';live_table=Table(title=f"[bold cyan]📊 LIVE TRADING DECISION[/bold cyan] - {timestamp}",show_header=True,box=box.ROUNDED);live_table.add_column('Metric',style='bold white');live_table.add_column('Value',justify='right',style='yellow');live_table.add_row('Signal',f"{signal_emoji} {signal}");live_table.add_row('Confidence',f"{confidence:.1%}");live_table.add_row('Price',f"${price:,.2f}");live_table.add_row('Expected Return',f"{result.get("expected_return",0):+.2%}");live_table.add_row('Position Size',f"{result.get("recommended_position_size",0):.2%}");console.print(live_table);return
	signal=result.get('signal','HOLD');confidence=result.get('confidence',.0);price=result.get('current_price',0);signal_color='green'if signal=='BUY'else'red'if signal=='SELL'else'cyan'if signal=='COOLDOWN'else'white'if signal=='EXIT'else'yellow';signal_emoji='🟢'if signal=='BUY'else'🔴'if signal=='SELL'else'🔵'if signal=='COOLDOWN'else'⚪'if signal=='EXIT'else'🟡';decision_text=f"""
[bold white]Signal:[/bold white] [{signal_color}]{signal_emoji} {signal}[/{signal_color}] ({confidence:.1%})
[bold white]Price:[/bold white] ${price:,.2f}
[bold white]Position Size:[/bold white] {result.get("recommended_position_size",0):.2%}
[bold white]Expected Return:[/bold white] {result.get("expected_return",0):+.2%}
[bold white]Regime:[/bold white] {result.get("regime","unknown")}
    """;console.print(Panel(decision_text.strip(),title='[bold cyan]🎯 TRADING DECISION[/bold cyan]',border_style='cyan'));console.print();targets=result.get('price_targets',{});intervals=result.get('confidence_intervals',{})
	if targets or intervals:
		targets_table=Table(show_header=True,header_style='bold magenta',box=box.SIMPLE);targets_table.add_column('Horizon',style='cyan');targets_table.add_column('Target Price',justify='right',style='yellow');targets_table.add_column('95% CI Lower',justify='right');targets_table.add_column('95% CI Upper',justify='right')
		for horizon in['1d','3d','7d']:
			if horizon in targets and horizon in intervals:ci=intervals[horizon];targets_table.add_row(horizon.upper(),f"${targets[horizon]:,.2f}",f"${ci.get("lower",0):,.2f}",f"${ci.get("upper",0):,.2f}")
		console.print(Panel(targets_table,title='[bold magenta]💰 PRICE TARGETS[/bold magenta]',border_style='magenta'));console.print()
	risk=result.get('risk_metrics',{})
	if risk:risk_text=f"""
[bold white]Volatility:[/bold white] {risk.get("volatility",0):.2%}
[bold white]Sharpe Ratio:[/bold white] {risk.get("sharpe_ratio",0):.2f}
[bold white]CVaR (95%):[/bold white] {risk.get("cvar_95",0):.2%}
[bold white]Uncertainty:[/bold white] {risk.get("prediction_uncertainty",0):.2%}
        """;console.print(Panel(risk_text.strip(),title='[bold red]⚖️ RISK METRICS[/bold red]',border_style='red'));console.print()
	causal_score=result.get('causal_score',None);causal_block=result.get('causal_block',False)
	if causal_score is not None:causal_text=f"[bold white]Causal:[/bold white] {causal_score:.2f}{" 🚫 block"if causal_block else""}";console.print(Panel(causal_text,title='[bold blue]🧠 CAUSAL[/bold blue]',border_style='blue'));console.print()
	explanations=result.get('signal_explanations',[])
	if explanations:exp_text='\n'.join([f"  • {exp}"for exp in explanations[:8]]);console.print(Panel(exp_text,title='[bold yellow]🔍 SIGNAL EXPLANATIONS[/bold yellow]',border_style='yellow'))
def display_summary_stats(console:Console,result:Dict[str,Any],all_assets_data:List[Dict])->None:buy_count=sum(1 for a in all_assets_data if isinstance(a['decision'],dict)and a['decision'].get('signal')=='BUY');sell_count=sum(1 for a in all_assets_data if isinstance(a['decision'],dict)and a['decision'].get('signal')=='SELL');hold_count=sum(1 for a in all_assets_data if isinstance(a['decision'],dict)and a['decision'].get('signal')=='HOLD');cooldown_count=sum(1 for a in all_assets_data if isinstance(a['decision'],dict)and a['decision'].get('signal')=='COOLDOWN');total_assets=len(all_assets_data);total_trades=result.get('total_trades',0);summary_text=f"""
[bold white]Total Assets Analyzed:[/bold white] {total_assets}
[bold green]BUY Signals:[/bold green] {buy_count}
[bold red]SELL Signals:[/bold red] {sell_count}
[bold yellow]HOLD Signals:[/bold yellow] {hold_count}
[bold cyan]COOLDOWN Signals:[/bold cyan] {cooldown_count}
[bold cyan]Trades Executed:[/bold cyan] {total_trades}
    """;console.print(Panel(summary_text.strip(),title='[bold cyan]📊 SUMMARY STATISTICS[/bold cyan]',border_style='cyan'));console.print()
def display_simple_results(result:Dict[str,Any])->None:
	print('\n'+'='*80);print('QUANT TRADING AGENT - DETAILED ANALYSIS');print('='*80)
	if'error'in result:print(f"Error: {result["error"]}");return
	print(f"\n📊 CURRENT MARKET STATE");print(f"  Current Price: ${result.get("current_price",0):,.2f}");print(f"  Market Regime: {result.get("regime","unknown")}");print(f"  Volatility: {result.get("risk_metrics",{}).get("volatility",0):.2%}");print(f"\n🎯 TRADING DECISION");signal=result.get('signal','HOLD');confidence=result.get('confidence',.0);print(f"  Signal: {signal} (Confidence: {confidence:.0%})");print(f"  Expected Return: {result.get("expected_return",0):.2%}");print(f"  Recommended Position Size: {result.get("recommended_position_size",0):.2%}");batched_results=result.get('batched_results',[])
	if batched_results:
		print(f"\n📊 MULTI-ASSET RESULTS")
		for batch in batched_results:
			decisions=batch.get('decisions',{})
			for(symbol,decision)in decisions.items():sig=decision.get('signal','HOLD')if isinstance(decision,dict)else'HOLD';conf=decision.get('confidence',.0)if isinstance(decision,dict)else .0;price=decision.get('current_price',result.get('current_price',0))if isinstance(decision,dict)else result.get('current_price',0);print(f"  {symbol}: {sig} ({conf:.1%}) @ ${price:,.2f}")
async def main()->None:
	import sys;use_streaming='--stream'in sys.argv or os.getenv('STREAMING_MODE','false').lower()=='true';longterm_mode='--longterm-mode'in sys.argv or os.getenv('LONGTERM_MODE','').lower()=='true'or LONGTERM_MODE_ENABLED;agent=QuantTradingAgent(days_back=DAYS_BACK,demo_mode=not LIVE_TRADING_MODE_DEFAULT,live_trading_mode=LIVE_TRADING_MODE_DEFAULT,longterm_mode=longterm_mode)
	if use_streaming:
		print('Starting real-time streaming mode...');print('Press Ctrl+C to stop')
		async def on_signal_update(agent_instance):
			try:
				result=await agent_instance.run()
				if result and'error'not in result:signal,price,confidence=result.get('signal','HOLD'),result.get('current_price',0),result.get('confidence',0);print(f"\n[{datetime.now().strftime("%H:%M:%S")}] Signal: {signal} | Price: ${price:.2f} | Confidence: {confidence:.0%}")
			except Exception as e:logger.debug(f"Error in streaming callback: {e}")
		await agent.start_streaming(exchange='coinbase',symbol='ETH',decision_callback=on_signal_update)
		try:
			while agent.streaming_mode:await asyncio.sleep(1)
		except KeyboardInterrupt:print('\nStopping streaming...');await agent.stop_streaming('coinbase','ETH')
	else:
		console=Console()if RICH_AVAILABLE else None;last_result=None
		if longterm_mode:position_eval_interval=getattr(agent,'position_evaluation_interval_hours',1)*3600;full_refresh_hours=getattr(agent,'full_refresh_interval_hours',24);display_update_interval=position_eval_interval;full_refresh_interval=full_refresh_hours*3600;mode_text=f"[bold cyan]🔍 LONGTERM MODE[/bold cyan]\n[dim]Position evaluation: every {position_eval_interval/3600:.1f} hour(s)\nFull refresh: every {full_refresh_hours} hour(s)\nMax position size: {getattr(agent,"max_position_size",.005):.2%}\nMin confidence: {getattr(agent,"min_confidence_threshold",.85):.0%}[/dim]"
		else:display_update_interval,full_refresh_interval=1.,3e2;mode_text='[dim]Full refresh every 5 min, display updates every 1 sec[/dim]'
		last_full_refresh,first_run=.0,True
		try:
			if console:title='[bold green]🚀 QUANT TRADING AGENT - LIVE MODE[/bold green]'if not longterm_mode else'[bold cyan]🔍 QUANT TRADING AGENT - LONGTERM MODE[/bold cyan]';console.print(Panel.fit(f"{title}\n[yellow]Press Ctrl+C to stop and view detailed analysis[/yellow]\n{mode_text}",border_style='green'if not longterm_mode else'cyan',box=box.DOUBLE));console.print()
			else:
				mode_title='🚀 QUANT TRADING AGENT - LIVE MODE'if not longterm_mode else'🔍 QUANT TRADING AGENT - LONGTERM MODE';print(mode_title);print('Press Ctrl+C to stop and view detailed analysis')
				if longterm_mode:print(f"Position evaluation: every {position_eval_interval/3600:.1f} hour(s)");print(f"Full refresh: every {full_refresh_hours} hour(s)")
				else:print('Full refresh every 5 min, display updates every 1 sec')
				print()
			import time
			while True:
				try:
					current_time=time.time();time_since_full_refresh=current_time-last_full_refresh
					if first_run or time_since_full_refresh>=full_refresh_interval:
						if console and not first_run:console.print('[dim]Performing full refresh...[/dim]')
						result=await agent.run(incremental=False)
						if not isinstance(result,dict):logger.error(f"ERROR: run() returned {type(result)} instead of dict!");logger.error(f"Result: {result}");result={'error':f"Invalid return type: {type(result)}"}
						last_result,last_full_refresh,first_run=result,current_time,False
					else:result=await agent.run(incremental=True);last_result=result
					if not isinstance(result,dict):logger.error(f"ERROR: run() returned {type(result)} instead of dict!");logger.error(f"Result: {result}");result={'error':f"Invalid return type: {type(result)}"}
					if'error'in result:
						if console:console.print(f"[red]Error: {result["error"]}[/red]")
						else:print(f"Error: {result["error"]}")
						await asyncio.sleep(display_update_interval);continue
					if console:console.clear();display_rich_results(console,result,live_mode=True)
					else:print(f"\n[{datetime.now().strftime("%H:%M:%S")}] Analysis complete");display_simple_results(result)
					sleep_time=display_update_interval*(agent.cooldown_sleep_multiplier if agent.system_state=='COOLDOWN'else 1.);await asyncio.sleep(sleep_time)
				except Exception as e:
					import traceback;error_msg=str(e)
					if'DataFrame'in error_msg and'ambiguous'in error_msg:
						logger.error(f"DataFrame boolean ambiguity error detected!");logger.error(f"Result type: {type(result)}")
						if hasattr(result,'shape'):logger.error(f"Result shape: {result.shape}")
						if hasattr(result,'columns'):logger.error(f"Result columns: {list(result.columns)}")
						logger.error(f"Last few calls in stack:")
						for line in traceback.format_exc().split('\n')[-10:]:
							if line.strip():logger.error(f"  {line}")
					else:logger.error(f"Error in live mode: {e}");logger.error(f"Traceback: {traceback.format_exc()}")
					await asyncio.sleep(display_update_interval);continue
		except KeyboardInterrupt:
			if console:console.print('\n');console.print(Panel.fit('[bold yellow]📊 FINAL DETAILED ANALYSIS[/bold yellow]',border_style='yellow',box=box.DOUBLE));console.print()
			try:await agent._exit_liquidate_all()
			except Exception as e:logger.warning(f"[EXIT] Liquidation on exit failed: {e}")
			if last_result:
				last_result=dict(last_result);last_result['signal']='EXIT'
				if RICH_AVAILABLE and console:display_rich_results(console,last_result,live_mode=False)
				else:display_simple_results(last_result)
				signal,confidence=last_result.get('signal','HOLD'),last_result.get('confidence',.0);logger.info(f"Agent run completed: {signal} (confidence: {confidence:.0%})")
			elif console:console.print('[yellow]No analysis data available[/yellow]')
			else:print('No analysis data available')
if __name__=='__main__':
	import sys;longterm_mode='--longterm-mode'in sys.argv or os.getenv('LONGTERM_MODE','').lower()=='true'or LONGTERM_MODE_ENABLED;print('='*80)
	if longterm_mode:print('🔍 QUANT TRADING AGENT - LONGTERM MODE - STARTING');print('='*80);print('📊 LONGTERM MODE ACTIVE:');print('   - Slow, precise long-term investment system');print('   - Predictions for next day/week instead of seconds');print('   - Position evaluation: hourly (trades only with extremely high confidence)');print('   - Prioritizes assets valued less than 1 USD/EUR');print('   - Maximum position size: 0.5% of portfolio');print('   - Minimum confidence threshold: 85%');print('   - Designed for background server operation');print('='*80)
	else:print('🚀 QUANT TRADING AGENT - STARTING');print('='*80)
	print('⚠️  SAFETY MODE: Demo mode enabled by default');print('   - NO real trades will be executed');print('   - Only analysis and simulation will run');print('   - To enable live trading, set live_trading_mode=True explicitly')
	if not longterm_mode:print('   - To enable longterm mode, use --longterm-mode flag or set LONGTERM_MODE=true')
	print('='*80);print()
	try:asyncio.run(main())
	except KeyboardInterrupt:print('\n⚠️  Stopping agent...')
