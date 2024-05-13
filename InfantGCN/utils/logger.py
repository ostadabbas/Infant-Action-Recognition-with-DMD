import logging

class Logger:
    def __init__(self, name, log_file):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG) 
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

    def log_message(self, message):
        self.logger.info(message)

    def log_training(self, epoch, loss, accuracy, lr):
        self.logger.info(f"Train: Epoch: {epoch}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}, Learning Rate: {lr:.5f}")

    def log_validation(self, epoch, loss, accuracy):
        self.logger.info(f"Validation: Epoch: {epoch}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

    def log_test(self, epoch, loss, accuracy):
        self.logger.info(f"Test: Epoch: {epoch}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")