import { Download, Video, Zap } from 'lucide-react';
import Button from '../ui/Button';
import homescreen from '../../assets/homescreen.png';
import translationscreen from '../../assets/translationscreen.png';
import { Link } from 'react-router-dom';

const Hero = () => {
  return (
    <section className="pt-32 pb-20">
      <div className="container mx-auto px-4 grid md:grid-cols-2 gap-12 items-center">
        <div>
          <h1 className="text-4xl md:text-5xl font-bold mb-6">
            Breaking Barriers in <span className="text-teal">Sign Language</span> Communication
          </h1>
          <p className="text-lg text-gray-600 mb-8">
            SynaEra translates sign language to text and speech in real-time, making communication accessible for everyone.
          </p>
          <div className="flex flex-col sm:flex-row gap-4">
            <Button variant="primary">
              <Download size={20} />
              <span>Download App</span>
            </Button>
            <Button variant="secondary">
              <Video size={20} />
              <span>Watch Demo</span>
            </Button>
            <Link to="/test-feature">
            <Button variant="outline">
              <Zap size={20} />
              <span>Test It</span>
            </Button>
            </Link>
          </div>
        </div>
        
        <div className="relative">
          <div className="bg-gradient-to-br from-teal/20 to-orange/20 rounded-3xl p-4">
            <div className="grid grid-cols-2 gap-4">
              <div className="bg-white rounded-2xl shadow-lg overflow-hidden">
                <img 
                  src={homescreen} 
                  alt="SynaEra App Home Screen" 
                  className="w-full h-auto"
                />
              </div>
              <div className="bg-white rounded-2xl shadow-lg overflow-hidden">
                <img 
                  src={translationscreen} 
                  alt="SynaEra Translation Screen" 
                  className="w-full h-auto"
                />
              </div>
            </div>
          </div>
          
          <div className="absolute -bottom-6 -right-6 bg-white p-3 rounded-xl shadow-lg">
            <div className="flex items-center gap-2">
              <div className="bg-green-500 h-3 w-3 rounded-full"></div>
              <span className="text-sm font-medium">Real-time Translation</span>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default Hero;