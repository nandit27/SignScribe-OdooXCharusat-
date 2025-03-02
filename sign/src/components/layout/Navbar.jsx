import { useState } from 'react';
import { Menu, X, Download } from 'lucide-react';
import Button from '../ui/Button';
import SignScribe from '../../assets/SignScribe.png';

const Navbar = () => {
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  
  return (
    <header className="fixed w-full bg-white/80 backdrop-blur-sm z-50 shadow-sm">
      <div className="container mx-auto flex items-center justify-between p-4">
        <div className="flex items-center gap-2">
          <img src={SignScribe} alt="SynaEra Logo" className="h-10" />
          <span className="text-xl font-semibold text-teal">SynaEra</span>
        </div>
        
        {/* Desktop Navigation */}
        <nav className="hidden md:flex items-center gap-8">
          <a href="#features" className="text-gray-600 hover:text-teal transition-colors">Features</a>
          <a href="#how-it-works" className="text-gray-600 hover:text-teal transition-colors">How It Works</a>
          <a href="#testimonials" className="text-gray-600 hover:text-teal transition-colors">Testimonials</a>
        </nav>
        
        <div className="hidden md:block">
          <Button variant="primary">
            <Download size={18} />
            <span>Download App</span>
          </Button>
        </div>
        
        {/* Mobile Menu Button */}
        <button 
          className="md:hidden text-gray-700"
          onClick={() => setIsMenuOpen(!isMenuOpen)}
        >
          {isMenuOpen ? <X size={24} /> : <Menu size={24} />}
        </button>
      </div>
      
      {/* Mobile Navigation */}
      {isMenuOpen && (
        <div className="md:hidden bg-white border-t">
          <div className="container mx-auto py-4 px-4 flex flex-col gap-4">
            <a 
              href="#features" 
              className="text-gray-600 hover:text-teal py-2 transition-colors"
              onClick={() => setIsMenuOpen(false)}
            >
              Features
            </a>
            <a 
              href="#how-it-works" 
              className="text-gray-600 hover:text-teal py-2 transition-colors"
              onClick={() => setIsMenuOpen(false)}
            >
              How It Works
            </a>
            <a 
              href="#testimonials" 
              className="text-gray-600 hover:text-teal py-2 transition-colors"
              onClick={() => setIsMenuOpen(false)}
            >
              Testimonials
            </a>
            <Button variant="primary" className="mt-2">
              <Download size={18} />
              <span>Download App</span>
            </Button>
          </div>
        </div>
      )}
    </header>
  );
};

export default Navbar;