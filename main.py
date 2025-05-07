import json
import logging
import os
import ssl
import time
from dotenv import load_dotenv
from kafka import KafkaConsumer

from course_generator import generate_course_content

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Kafka configuration
CONSUMER_GROUP = os.getenv('KAFKA_CONSUMER_GROUP', 'course_generator_group')
KAFKA_BROKERS = os.getenv('KAFKA_BROKERS', 'localhost:9092')
KAFKA_TOPIC = os.getenv('KAFKA_TOPIC', 'course_topics')
KAFKA_USERNAME = os.getenv('KAFKA_USERNAME', '')
KAFKA_PASSWORD = os.getenv('KAFKA_PASSWORD', '')


def create_consumer(brokers, topic, username, password):
    """
    Create a Kafka consumer with the specified configuration.
    
    Args:
        brokers (str): Comma-separated list of Kafka broker addresses
        topic (str): Kafka topic to subscribe to
        username (str): SASL username for authentication
        password (str): SASL password for authentication
        
    Returns:
        KafkaConsumer: Configured Kafka consumer instance
    """
    try:
        # Create an SSL context that does not verify the certificate
        context = ssl.create_default_context()
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE

        return KafkaConsumer(
            topic,
            bootstrap_servers=brokers,
            security_protocol='SASL_SSL',
            sasl_mechanism='SCRAM-SHA-512',
            sasl_plain_username=username,
            sasl_plain_password=password,
            ssl_context=context,
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            auto_offset_reset='latest',
            enable_auto_commit=False,
            fetch_max_bytes=1048576,
            max_partition_fetch_bytes=1048576,
            group_id=CONSUMER_GROUP
        )
    except Exception as e:
        logger.error(f"Error creating Kafka consumer: {e}")
        raise


def process_message(message):
    """
    Process a message from Kafka.
    
    Args:
        message: Kafka message containing course structure data
    """
    try:
        logger.info(f"Received message: {message}")
        
        # Extract course data from the message
        course_data = message.value
        
        # Validate message structure
        if not all(key in course_data for key in ['title', 'modules']):
            logger.error(f"Invalid message format. Required fields missing: {course_data}")
            return
            
        # Generate course content
        course_content = generate_course_content(course_data)
        
        # Save the generated course content
        save_course_content(course_data['title'], course_content)
        
        logger.info(f"Successfully processed course: {course_data['title']}")
        
    except Exception as e:
        logger.error(f"Error processing message: {e}")


def save_course_content(title, content):
    """
    Save the generated course content to a markdown file.
    
    Args:
        title (str): Course title
        content (str): Generated course content in markdown format
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs('generated_courses', exist_ok=True)
        
        # Sanitize filename
        filename = title.lower().replace(' ', '_').replace('/', '_').replace('\\', '_')
        filepath = f"generated_courses/{filename}.md"
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
            
        logger.info(f"Course content saved to {filepath}")
    except Exception as e:
        logger.error(f"Error saving course content: {e}")


def main():
    """Main function to run the Kafka consumer."""
    logger.info("Starting Kafka consumer for course generation...")
    
    try:
        consumer = create_consumer(
            KAFKA_BROKERS,
            KAFKA_TOPIC,
            KAFKA_USERNAME,
            KAFKA_PASSWORD
        )
        
        logger.info(f"Consumer created and subscribed to topic: {KAFKA_TOPIC}")
        
        # Consume messages
        for message in consumer:
            process_message(message)
            
            # Commit the offset
            consumer.commit()
            
    except KeyboardInterrupt:
        logger.info("Consumer stopped by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    finally:
        if 'consumer' in locals():
            consumer.close()
            logger.info("Consumer closed")


if __name__ == "__main__":
    main()