/* Making Sense of 'this' by Using Arrow Functions, That and Bind */

console.log(this); // Global environment

let myObject = {
  whatIsThis: function() {
    console.log(this);
  }
};
myObject.whatIsThis(); // Object

let myObject2 = {
  name: 'Iron Man',
  whatIsThis: function() {
    console.log(name); // will not work
  },
}
// myObject2.whatIsThis(); // undefined

let myObject3 = {
  name: 'Iron Man',
  whatIsThis: function() {
    console.log(this.name); // will not work
  },
}
myObject3.whatIsThis();

class Fruit {
  constructor(name) {
    this.name = name;
  }
  getName() {
    return this.name;
  }
}
let apple = new Fruit('Apple');
console.log(apple.getName());

let orange = new Fruit('Orange');
console.log(orange.getName());

// When it can break down
let counter = {
  initial: 100,
  interval: 1000,
  startCounting: function() {
    console.log('startCounting');
    console.log(this);
    setInterval(function() {
      console.log('setInterval:');
      console.log(this);
      this.initial++;
      console.log(this.initial)
    }, this.interval);
  },
}
// counter.startCounting();
/* The anonymous function inside the setInterval does not get created when the counter object is initialised but only the startCounting method is called. Thus it is in the context of the window object and only the initial value of the window object is available; leading to NaN values.*/

// Fix One - declare local variable
let counter2 = {
  initial: 100,
  interval: 1000,
  startCounting: function() {
    // Declare a local variable to store proper this reference
    let that = this;
    setInterval(function() {
      that.initial++;
      console.log(that.initial);
    }, this.interval);
  },
}
counter2.startCounting();

// The variable can be any name
let counter3 = {
  initial: 100,
  interval: 1000,
  startCounting: function() {
    // Declare a local variable to store proper this reference
    let baconAndEggs = this;
    setInterval(function() {
      baconAndEggs.initial++;
      console.log(baconAndEggs.initial);
    }, this.interval);
  },
}
counter3.startCounting();

// Fix Two - use arrow function
let counter4 = {
  initial: 100,
  interval: 1000,
  startCounting: function() {
    // Use arrow function
    setInterval(() => {
      this.initial++;
      console.log(this.initial);
    }, this.interval);
  },
}
counter4.startCounting();
/* Arrow functions inherit this value from their surroundings and also lack constructor or prototype properties. They do not support bind, call and apply methods. */

// Fix Three - use the bind method
/*
let boundFunction = myRegularFunction.bind(valueOfThis);
boundFunction();
*/
let counter5 = {
  initial: 100,
  interval: 1000,
  startCounting: function() {
    // Use bind method
    setInterval(function() {
      this.initial++;
      console.log(this.initial);
    }.bind(this), this.interval);
  },
}
counter5.startCounting();
