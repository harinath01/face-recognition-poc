import { render } from 'solid-js/web'
import './style.css'
import { App } from './App'

render(() => <App />, document.querySelector<HTMLDivElement>('#app')!)
