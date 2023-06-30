/* The JavaScript Math object contains some really useful and powerful mathematical operations that can be used in web development, but it lacks many important operations that most other languages provide (such as Haskell, which has a huge number of them).
https://www.sitepoint.com/javascript-missing-math-methods/
*/

// Missing Math Methods in JavaScript: Sum
function jsSum(array){
  return array.reduce((sum, number) => sum + number, 0);
}

// Missing Math Methods in JavaScript: Product
function jsProduct(array){
  return array.reduce((total, num) => total*num, 1);
}

// Missing Math Methods in JavaScript: Odd and Even
function jsEven(number){
  return number % 2 === 0;
}

function jsOdd(number){
  return number % 2 !== 0;
}

function checkWinner(gamesPlayed){
  let winner
  if(jsOdd(gamesPlayed)){
      winner = "player1";
  }
  else{
      winner = "player2";
  }
  return winner;
}

// Missing Math Methods in JavaScript: triangleNumber
function jsTriangleNumber(number){
  return 0.5 * number * (number + 1);
}

// Missing Math Methods in JavaScript: Factorial
function jsFactorial(number){
  if (number <= 0){
    return 1;
  }
  else{
    return number * jsFactorial(number - 1);
  }
}

// Missing Math Methods in JavaScript: Factors
function jsFactors(number){
  let factorsList = [];
  for(let count = 1; count < number + 1; count++){
      if(number % count === 0){
          factorsList.push(count);
      }
  }
  return factorsList;
}

function createTeams(numberOfPlayers, numberOfTeams){
  let playersInEachTeam;
  if(jsFactors(numberOfPlayers).includes(numberOfTeams)){
      playersInEachTeam = numberOfPlayers / numberOfTeams;
  }
  else{
      playersInEachTeam = "wait for more players";
  }
  return playersInEachTeam;
}

// Missing Math Methods in JavaScript: isPrime
function jsIsPrime(number){
  return jsFactors(number).length === 2;
}

function addUsers(users){
  if(jsIsPrime(users)){
      wait = true;
  }
  else{
      wait = false;
  }
}

// Missing Math Methods in JavaScript: gcd (Greatest Common Divisor) or HCF (Highest Common Factor)
function jsGcd(number1, number2){
  let inCommon = [];
  for(let i of jsFactors(number1)){
      if(jsFactors(number2).includes(i)){
          inCommon.push(i);
      }
  }
  return inCommon.sort((a,b)=> b - a)[0];
}

function gcd(number1, number2){
  if(number2 === 0){
      return number1;
  }
  else{
      return gcd(number2, number1 % number2);
  }
}

// Missing Math Methods in JavaScript: lcm (Lowest Common Multiple)
/* However, there’s a very useful formula that we can use to calculate the lowest common multiple:

(number1 x number2) / the Greatest Common Divisor of the two numbers
 */

function jsLcm(number1, number2){
  return (number1 * number2) / gcd(number1, number2);
}

