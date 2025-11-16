import React from 'react';
import './App.css';
import DamageAssessment from './components/DamageAssessment';
import ThemeToggle from './components/ThemeToggle';
import RedlineLogo from './components/RedlineLogo';

function App() {
  return (
    <div className="App">
      <header className="App-header">
        <div className="header-content">
          <div>
            <RedlineLogo />
            <p className="subtitle">AI-Powered Vehicle Damage Assessment</p>
          </div>
          <ThemeToggle />
        </div>
      </header>
      <main>
        <DamageAssessment />
      </main>
      <footer>
        <p>Upload a vehicle image to get instant damage assessment and cost estimates</p>
      </footer>
    </div>
  );
}

export default App;

