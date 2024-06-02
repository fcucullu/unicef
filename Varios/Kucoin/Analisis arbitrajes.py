############################################################################
'''                           ANALISIS DE ARBITRAJES                     '''
############################################################################

import pandas as pd
pd.set_option('display.float_format', '{:.12f}'.format)
pd.options.mode.chained_assignment = None


############################################################################
'''                                Objetos                               '''
############################################################################

#Creo un objeto PLAZA que tendra atributos como la caja de puntas y otros
class Plaza():
    def __init__(self, pair, price_currency, base_currency):        
        self.price = price_currency
        self.base = base_currency
        self.caja = pd.DataFrame({'Hora': [0.0,0.0,0.0,0.0,0.0],
                                  'QC': [0.0,0.0,0.0,0.0,0.0],
                                  'PC': [0.0,0.0,0.0,0.0,0.0],
                                  'PV': [0.0,0.0,0.0,0.0,0.0],
                                  'QV': [0.0,0.0,0.0,0.0,0.0]
                                  })
    
    #Este metodo resetea la caja
    def reset_caja(self):
        self.caja = pd.DataFrame({'Hora': [0.0,0.0,0.0,0.0,0.0],
                                  'QC': [0.0,0.0,0.0,0.0,0.0],
                                  'PC': [0.0,0.0,0.0,0.0,0.0],
                                  'PV': [0.0,0.0,0.0,0.0,0.0],
                                  'QV': [0.0,0.0,0.0,0.0,0.0]
                                  })
        
#Creo el objeto EXCHANGE que tendrá varias plazas y una estructura de comisiones particular    
class Exchange():
    def __init__(self, exchange, fees):#, exchange, price_currency, base_currency):
        self.exchange = exchange
        try:
            self.fee = fees[exchange]
        #Capturo la excepcion en donde el usuario no introduce las comisiones
        except:
            pass
    


############################################################################
'''                           Funciones                                '''
############################################################################


def arbitraje_directo(exchange_origin, exchange_comparison, par, trades):
    ''' Esta funcion analiza existencia de arbitrajes entre distintas exchanges
        comparando plazas de exactamente el mismo par.
        No analiza arbitrajes trianguales ni pares invertidos (RIF/BTC vs BTC/RIF).
        
        Inputs:
            exchange_origin: objeto del tipo "exchange"
            exchange_comparison: objeto del tipo "exchange"
            par: ticker del instrumento en formato string sin signos (BTC/USD->BTCUSD)
            trades: pandas.DataFrame en donde se guardarán los resultados
        
        Output:
            Retorna la tabla "trades" con la información incorporada 
        
    '''
    
    caja_origin = getattr(exchange_origin, par).caja
    fee_origin = exchange_origin.fee
    caja_comparison = getattr(exchange_comparison, par).caja
    fee_comparison = exchange_comparison.fee 
    
    side = 'C'
    depth_origin = 0
    depth_comparison = 0
    q_remanente = caja_origin['QV'][depth_origin]
    
    while depth_origin < len(caja_origin) and depth_comparison < len(caja_comparison) and (caja_origin['PV'][depth_origin] * (1 + fee_origin)) < (caja_comparison['PC'][depth_comparison] * (1 - fee_comparison)) and caja_origin['PV'][depth_origin] != 0:

        hora = max(caja_origin['Hora'][depth_origin], caja_comparison['Hora'][depth_comparison])
        #print(hora)
        #print('Comprar {} a {} y vender a {}'.format(par, caja_origin['PV'][depth_origin] * (1 + fee_origin), caja_comparison['PC'][depth_comparison] * (1 + fee_comparison)))
        
        if side == 'C':
                
            if q_remanente > caja_comparison['QC'][depth_comparison]:
                
                q_remanente -= caja_comparison['QC'][depth_comparison]
                q_ejecutada = caja_comparison['QC'][depth_comparison]
                #side = 'C'
                        
                trades = registrar_trade(trades, hora, caja_origin['PV'][depth_origin],
                          caja_comparison['PC'][depth_comparison], q_ejecutada,
                          exchange_origin, exchange_comparison, par)
                
                depth_comparison += 1
                
            elif q_remanente < caja_comparison['QC'][depth_comparison]:
                
                side = 'V'
                q_ejecutada = q_remanente
                q_remanente = caja_comparison['QC'][depth_comparison] - q_remanente
                
                trades = registrar_trade(trades, hora, caja_origin['PV'][depth_origin],
                          caja_comparison['PC'][depth_comparison], q_ejecutada,
                          exchange_origin, exchange_comparison, par)
                
                depth_origin += 1
                
            elif q_remanente == caja_comparison['QC'][depth_comparison]:
                
                q_ejecutada = q_remanente
                q_remanente = caja_origin['QV'][depth_origin + 1]
                
                trades = registrar_trade(trades, hora, caja_origin['PV'][depth_origin],
                          caja_comparison['PC'][depth_comparison], q_ejecutada,
                          exchange_origin, exchange_comparison, par)
                
                depth_comparison += 1
                depth_origin += 1
                
        elif side == 'V':
            
            if q_remanente > caja_origin['QV'][depth_origin]:
                
                q_remanente -= caja_origin['QV'][depth_origin]
                q_ejecutada = caja_origin['QV'][depth_origin]
                #side = 'V'
                
                trades = registrar_trade(trades, hora, caja_origin['PV'][depth_origin],
                          caja_comparison['PC'][depth_comparison], q_ejecutada,
                          exchange_origin, exchange_comparison, par)
                
                depth_origin += 1
                
            elif q_remanente < caja_origin['QV'][depth_origin]:
                
                side = 'C'
                q_ejecutada = q_remanente
                q_remanente = caja_origin['QV'][depth_origin] - q_ejecutada
                
                trades = registrar_trade(trades, hora, caja_origin['PV'][depth_origin],
                          caja_comparison['PC'][depth_comparison], q_ejecutada,
                          exchange_origin, exchange_comparison, par)
                
                depth_comparison += 1
                
            elif caja_origin['QV'][depth_origin] == q_remanente:
                
                q_ejecutada = q_remanente
                q_remanente = caja_origin['QV'][depth_origin + 1]
                
                trades = registrar_trade(trades, hora, caja_origin['PV'][depth_origin],
                          caja_comparison['PC'][depth_comparison], q_ejecutada,
                          exchange_origin, exchange_comparison, par)
                
                depth_origin +=1
                depth_comparison += 1
                
    return trades

#Esta funcion registrará todos los trades que se encuentren en un momento determinado
def registrar_trade(trades, hora, precio_compra, precio_venta, q_ejecutada,
              exchange_origin, exchange_comparison, par):
    
    ''' Esta función registra una operación en una pandas.Dataframe con las columnas:
        Hora / Exchange_Compradora / Exchange_Vendedora / Par / Cantidad / Precio_Compra / 
        Precio_Venta / Monto_invertido / Ganancia
        
        Inputs: 
            trades: es la pd.DataFrame en donde se guardará la información
            hora: string con la hora de la operación
            precio_compra, precio_venta, q_ejecutada: floats
            exchange_origin, exchange_comparison: objetos del tipo exchange
            par: ticker del par en formato string
            
            Output:
                Retorna la tabla "trades" con la información incorporada 
    '''
    
    fee_origin = exchange_origin.fee
    fee_comparison = exchange_comparison.fee 
    origin = exchange_origin.exchange
    comparison = exchange_comparison.exchange
    
    trades = trades.append({'Hora': hora,
                   'Exchange_Compradora': origin,
                   'Exchange_Vendedora': comparison,
                   'Par': par,
                   'Cantidad': q_ejecutada,
                   'Precio_Compra': precio_compra * (1 + fee_origin),
                   'Precio_Venta': precio_venta * (1 - fee_comparison),
                   'Monto_Invertido': precio_compra * q_ejecutada * (1 + fee_origin),
                   'Ganancia': ( precio_venta * (1 - fee_comparison) - precio_compra * (1 + fee_origin) ) * q_ejecutada
                   }, ignore_index=True)

    return trades

#Funcion para guardar archivo en ruta que declare el usuario
def guardar_tabla(ruta, tabla):
    '''
        Esta funcion guarda una pd.DataFran en la ruta que declare el usuario.
        
        Inputs:
            ruta: path válido para python
            tabla: pd.DataFrame
            
        Outputs:
            None.
            
    '''
    tabla.to_csv(ruta, index=False, sep=';', decimal=',',
                 columns=['Hora','Exchange_Compradora','Exchange_Vendedora','Par','Cantidad','Precio_Compra','Precio_Venta','Monto_Invertido','Ganancia'])  


def arbitraje_directo_masivo(trades, exchanges, pares):
    '''
        Esta función recorre todas los values de un diccionario para hallar
        de forma masiva la existencia de arbitrajes directos.
        
        Inputs:
            trades: pd.DataFrame para almacenar los resultados
            exchanges: diccionario que contiene en sus keys los nombres de las exchanges
                a analizar y objetos del tipo "exchange" en sus values.
            pares: lista de listas que contienen en su primer elemento el ticker del instrumento
                sin signos ( [ [BTCUSD, ...], [RIFUSD, ...] ])
              
        Output:
            Retorna la tabla "trades" con la información incorporada 
            
    ''' 
    for i in pares:
        par = i[0]  
        for exchange_origin in exchanges.values():  
            for exchange_comparison in exchanges.values():
                if exchange_comparison == exchange_origin: continue
            
                trades = arbitraje_directo(exchange_origin, exchange_comparison, par, trades)
                        
    return trades
     
# Funcion general/global que disparara todo el script
def screening(raw, exchanges, pares, ruta_destino):
    
    '''
    Esta funcion simula una rueda real. 
    Para ello, recorrera información histórica, actualizará los objetos de exchanges y
    reproducirá lo ocurrido en sus plazas durante las ruedas.
    Cuando detecte que hay un cambio en el horario, monitoreará la existencia de
    arbitrajes.
    
    Inputs:
        raw: pd.DataFrame con, al menos, las siguientes columnas:
            exchange: string con el exchange de la observacion
            depth: numero del 0 al infinito que informe que orden tiene la observacion en la caja de puntas
            pair: par de monedas separadas por un '/', por ejemplo RIF/BTC
            timestamp: es el horario de la observacion.
            amount y price: cantidad y precio de la observacion
            type: es el side de la observacion. Debe ser 'bids' o 'asks.
        exchanges: diccionario que contiene en sus keys los nombres de las exchanges
                a analizar y objetos del tipo "exchange" en sus values.
        pares: lista de listas que contienen en su primer elemento el ticker del instrumento
                sin signos ( [ [BTCUSD, ...], [RIFUSD, ...] ])
        ruta_destino: path válido para python
        
    Output:
        Retorna la tabla "trades" con la información incorporada 
                
    '''
    
    trades = pd.DataFrame()
    
    for index in range(len(raw)):
        if raw.iloc[index]['exchange'] in exchanges:
            
            pair = raw.iloc[index]['pair'].replace('/','')
            depth = raw.iloc[index]['depth']
            
            if depth < len(getattr(exchanges[raw.iloc[index]['exchange']], pair).caja):
                
                caja = getattr(exchanges[raw.iloc[index]['exchange']], pair).caja
                caja['Hora'][depth] = raw.iloc[index]['timestamp']
                
                if raw.iloc[index]['type'] == 'bids':
                    caja['QC'][depth] = raw.iloc[index]['amount']
                    caja['PC'][depth] = raw.iloc[index]['price']
                    
                    
                elif raw.iloc[index]['type'] == 'asks':
                    caja['PV'][depth] = raw.iloc[index]['price']
                    caja['QV'][depth] = raw.iloc[index]['amount']
                    
                
            else:
                pass
            

        if raw.iloc[index]['timestamp'] != raw.iloc[index+1]['timestamp']:
            trades = arbitraje_directo_masivo(trades, exchanges, pares)

    guardar_tabla(ruta_destino, trades)
                
    return trades








############################################################################
'''                           ANALISIS                                '''
############################################################################

''' EXCHANGES ANALIZAR Y SUS COMISIONES '''
fees = {
'liquid': 0.001, #https://www.liquid.com/fees/
'kucoin': 0.0006, #https://www.kucoin.com/news/en-kumex-fee
'coinbene': 0.001, #https://www.cryptowisser.com/exchange/coinbene/
'bithumb': 0.001, #https://www.bithumb.pro/en-us/fee  
'bitfinex': 0.002, #https://www.bitfinex.com/fees
'coinall': 0.0015 #https://www.coinall.com/en/fees.html
#Luego incluir las comisiones de deposito y exrtraccion!!!
    }


''' DATA '''
raw = pd.read_csv(r'C:\Users\Francisco\Desktop\Trabajo\XCapit\Kucoin\DATA REAL\order_books_22-02-2020.csv')
#divido el pair para tener price y base currencies mas a mano
raw[['Price Currency','Base Currency']] = raw.pair.str.split('/', expand=True)
#Confirmo visualmente que la data esta ordenada cronologicamente
raw['timestamp'] =  pd.to_datetime(raw['timestamp'], format='%Y-%m-%d %H:%M:%S.%f')
raw = raw.sort_values('timestamp').reset_index(drop=True)


''' CASTEO LAS PLAZAS '''
plazas = raw['pair'].unique()
pares = []
for par in plazas:
    a,b = par.split('/')
    pares.append([a+b,a,b])
del a, b, plazas

exchanges = fees.keys()
temp = dict()
for i in exchanges:
    temp[i] = Exchange(i, fees)
    for par in pares:
        setattr(temp[i], par[0], Plaza(par[0], par[1], par[2]))
exchanges = temp
del temp, i, par 

''' CORRO ANALISIS '''

ruta_destino = r'C:\Users\Francisco\Desktop\Analisis.csv'
trades = screening(raw, exchanges, pares, ruta_destino)

