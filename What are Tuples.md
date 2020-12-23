What are Tuples?
Picture a messy desk. Pencils are scattered everywhere, various stacks of paper covering each other with some even covering pencils. Some notebooks are placed haphazardly and in the center is a laptop. Sure, we could argue that it’s our desk and we know where everything is. But, the problem arises when someone else has to use this desk to do work! 🙀

In programming, we often share our code with other developers and work together to create an app, develop a feature, or simply solve a problem. Messy and hard to read code might work for only one person but not for two (or more) people. One way we can organize our code is to use data types like arrays, dictionaries, and sets.

Another data type we can use are tuples. Tuples group together values that are enclosed in parentheses and separated by commas. Unlike any of the previous data types mentioned, tuples can include different types of data! If we used a tuple to help organize our desk, we could store our favorite pencil and eraser together in a small container labeled writingUtensils. By storing them together, we can keep our desk clean and always have easy access to our items.

Here’s an example of a tuple:

var easyAsPie = ("easy as", 3.14)
Above, we created easyAsPie that holds 2 values: "easy as", and 3.14. If we wanted to access each value individually, we can use dot syntax along with the index of the value:

var easyAsPie = ("easy as", 3.14)
var firstValue = easyAsPie.0   // "easy as"
var secondValue = easyAsPie.1  // 3.14
We can also name a tuple’s elements by prepending each one with a name and a colon - similar to key-value pair syntax in dictionaries:

var easyAsPie = (saying: "easy as", amount: 3.14)
var firstElementValue = easyAsPie.saying  // "easy as"
var secondElementValue = easyAsPie.amount // 3.14
In both cases, we were able to access and retrieve the tuple’s values using dot syntax.

When Do We Use Tuples?
Tuples have a variety of use cases that are surfaced in different parts of the Swift language as well as in this course — we’ll focus on how they’re used with dictionaries.

In dictionaries, tuples are used to iterate over key-value pairs.

For example, take a look at the dictionary lionKing1994 that stores the characters and their associated actors from the 1994 movie, “The Lion King”:

var lionKing1994 = [
  "Simba": "Matthew Broderick",
  "Mufasa": "James Earl Jones",
  "Rafiki": "Robert Guillaume"
]
To iterate through both the keys and the values of this dictionary, we will need to use a tuple in a for-in loop:

for (character, actor) in lionKing1994 {
  print("\(character) was voiced by \(actor).")
}
In the code snippet above, we wrap the key placeholder, character, and value placeholder, actor, in the tuple (character, actor). With each iteration, we are able to extract each key and value from the dictionary and print the paired data in an organized manner:

Mufasa was voiced by James Earl Jones.
Rafiki was voiced by Robert Guillaume.
Simba was voiced by Matthew Broderick.
To learn more about looping through a dictionary with tuples, check out the Swift exercise Iterating Through a Dictionary.
