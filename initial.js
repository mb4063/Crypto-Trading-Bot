require('dotenv').config();

'use strict'

const { Console } = require('console')
const { WebsocketAPI } = require('@binance/connector')

const logger = new Console({ stdout: process.stdout, stderr: process.stderr })

const apiKey = process.env.BINANCE_API_KEY || ''
const apiSecret = process.env.BINANCE_API_SECRET || ''

const callbacks = {
  open: (client) => {
    logger.debug('Connected with Websocket server')
    client.avgPrice('BTCUSDT')
  },
  close: () => logger.debug('Disconnected with Websocket server'),
  message: data => logger.info(data)
}

const websocketAPIClient = new WebsocketAPI(apiKey, apiSecret, { logger, callbacks })

// disconnect after 20 seconds
setTimeout(() => websocketAPIClient.disconnect(), 20000)
