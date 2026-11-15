import { Link } from 'react-router-dom'
import logo from '../assets/logo_fdc1d6.jpg'

export default function Login() {
  return (
    <main className="page centered">
      <img className="logo" src={logo} alt="ASLingo" />
      <Link className="button" to="/menu">Login</Link>
      <Link className="button" to="/menu">Sign Up</Link>
    </main>
  )
}
