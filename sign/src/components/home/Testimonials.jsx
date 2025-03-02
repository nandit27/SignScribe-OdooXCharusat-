import { motion } from 'framer-motion';

const Testimonials = () => {
  const testimonials = [
    {
      quote: "SynaEra has transformed how I communicate with my deaf colleagues. It's intuitive and accurate.",
      author: "Sarah J.",
      role: "Teacher"
    },
    {
      quote: "As someone learning sign language, this app has been an invaluable tool for practice and verification.",
      author: "Michael T.",
      role: "Student"
    },
    {
      quote: "The real-time translation feature has made my daily interactions so much smoother and more natural.",
      author: "Elena R.",
      role: "Interpreter"
    }
  ];
  
  return (
    <section id="testimonials" className="py-20 bg-gray-50">
      <div className="container mx-auto px-4">
        <div className="text-center mb-16">
          <motion.h2 
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.5 }}
            className="text-3xl font-bold mb-4"
          >
            What Our Users Say
          </motion.h2>
          <motion.p
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.5, delay: 0.1 }}
            className="text-gray-600 max-w-2xl mx-auto"
          >
            Hear from people who use SynaEra in their daily lives
          </motion.p>
        </div>
        
        <div className="grid md:grid-cols-3 gap-8">
          {testimonials.map((testimonial, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5, delay: index * 0.1 }}
              className="bg-white p-6 rounded-xl shadow-sm"
            >
              <div className="text-orange-DEFAULT text-4xl font-serif mb-4">"</div>
              <p className="text-gray-700 mb-6">{testimonial.quote}</p>
              <div>
                <p className="font-semibold">{testimonial.author}</p>
                <p className="text-gray-500 text-sm">{testimonial.role}</p>
              </div>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default Testimonials;