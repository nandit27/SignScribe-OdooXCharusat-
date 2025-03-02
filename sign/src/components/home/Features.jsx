import { Video, MessageSquare, BookOpen, Smartphone } from 'lucide-react';

const Features = () => {
  const features = [
    {
      icon: <Video className="h-10 w-10 text-teal" />,
      title: "Real-time Translation",
      description: "Convert sign language to text and speech instantly through your device's camera."
    },
    {
      icon: <MessageSquare className="h-10 w-10 text-orange-DEFAULT" />,
      title: "Custom Gestures",
      description: "Add your own custom gestures for personalized communication needs."
    },
    {
      icon: <BookOpen className="h-10 w-10 text-teal" />,
      title: "Learning Resources",
      description: "Access a library of sign language tutorials categorized by topic."
    },
    {
      icon: <Smartphone className="h-10 w-10 text-orange-DEFAULT" />,
      title: "Offline Mode",
      description: "Use core features without an internet connection, anywhere you go."
    }
  ];
  
  return (
    <section id="features" className="py-20 bg-gray-50">
      <div className="container mx-auto px-4">
        <div className="text-center mb-16">
          <h2 className="text-3xl font-bold mb-4">
            Powerful Features
          </h2>
          <p className="text-gray-600 max-w-2xl mx-auto">
            SynaEra combines cutting-edge technology with intuitive design to make sign language communication seamless.
          </p>
        </div>
        
        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-8">
          {features.map((feature, index) => (
            <div
              key={index}
              className="bg-white p-6 rounded-xl shadow-sm hover:shadow-md transition-shadow"
            >
              <div className="mb-4">{feature.icon}</div>
              <h3 className="text-xl font-semibold mb-2">{feature.title}</h3>
              <p className="text-gray-600">{feature.description}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default Features;