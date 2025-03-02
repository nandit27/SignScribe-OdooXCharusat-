import Button from '../ui/Button';

const DownloadSection = () => {
  return (
    <section className="py-20">
      <div className="container mx-auto px-4">
        <div className="bg-gradient-to-r from-teal to-orange-DEFAULT rounded-2xl p-12 text-white text-center">
          <h2 className="text-3xl font-bold mb-4">Ready to Break Communication Barriers?</h2>
          <p className="max-w-2xl mx-auto mb-8">
            Download SynaEra today and experience seamless sign language translation at your fingertips.
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Button variant="white">
              <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="lucide lucide-apple">
                <path d="M12 20.94c1.5 0 2.75-.67 3.95-1.89a8.8 8.8 0 0 0 2.05-5.55c0-2.84-1.5-4.73-3.55-5.75-1-.5-2.2-.75-2.45-.75a4.05 4.05 0 0 0-2.45.75c-2.1 1.02-3.55 2.91-3.55 5.75 0 2.79 1.5 4.9 2.05 5.55 1.2 1.22 2.45 1.89 3.95 1.89Z"/>
                <path d="M15 6c0-2.21-1.5-4-3-4S9 3.79 9 6v.5c0 2.21 1.5 4 3 4s3-1.79 3-4z"/>
              </svg>
              <span>App Store</span>
            </Button>
            <Button variant="white">
              <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="lucide lucide-play">
                <polygon points="5 3 19 12 5 21 5 3"/>
              </svg>
              <span>Google Play</span>
            </Button>
          </div>
        </div>
      </div>
    </section>
  );
};

export default DownloadSection;