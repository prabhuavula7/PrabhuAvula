from pyspark import SparkContext
from pyspark.streaming import StreamingContext

# Create a local StreamingContext with a batch interval of 10 seconds
sc = SparkContext("yarn", "NetworkWordCount")
ssc = StreamingContext(sc, 10)

# Create a DStream
lines = ssc.socketTextStream("104.155.148.83", 3333)

# Split each line into words
words = lines.flatMap(lambda line: line.split(" "))

# Count each word in each batch
pairs = words.map(lambda word: (word, 1))
wordCounts = pairs.reduceByKey(lambda x, y: x + y)

# Print each batch
wordCounts.pprint()

ssc.start()             # Start the computation
ssc.awaitTermination()  # Wait for the computation to terminate

