class Worker implements Runable {
	public void run() {
		for (int i = 0; i < 1000; ++i) {
			doWork();
		}
	}
	void doWork() {
	}
}
