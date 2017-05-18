### refresh page
`command r`

### developer tool
`opt cmd i`
`opt cmd j`

### open webpage for the folder
method1:
- cd to the folder
- `http-server` and go to the prepared web address
method2:
- cd to the folder
- `atom .`
- package, http-server, start or stop

### breakpoint: Normal
- click a line in source code
- Continue: symbol or F8
- see and uncheck and check breakpoints in one place: right panel to source code
- right click and remove, deactivate all breakpoints: right panel to source code

### conditional breakpoint:
- click a line
- right click to set condition to break

### debug mode
- breakpoint
- Continue: pause at breakpoint, and move on to next breakpoint
- Step over: run line by line
- Step Into: into a function
- Step out: out of a function

### best practice
- only use `===` or `!==`
- not use `==` or `!=`
- use function to set local variable

### add watch mode
- watch a variable or an expression with a variable
- in source code, select a variable and right click, select 'add to Watch'
- or in watch panel, click '+', type name of a variable `myArray` or expression `myArray.length`
- and right click watch panel to remove all watches

### Scope panel
- when using breakpoint or watches, we can also check the scope environment and the objects in the environment using Scope
- local, closure, global scopes can be seen at the same time

### call stack panel
- trace the path of program
- level by level deeper, than level by level getting out

### console.log and clear
- `console.log("...")`: log some message
- `console.clear()`: clear previous logs then make new `console.log("...")` below

### console.assert
- useful to test certain conditions are met or not
```javascript
myArray=[];
console.assert(myArray.length >= 1, "length is not >= 1, but "+myArray.length)
```

### console.table
- how to print out table data nicely
```javascript
totalFunctionsCalled++;
$.getJSON( 'data/data.json', function( data ) {
  var items = [];
  var className = '';
  $.each( data, function(i) {
	className = (i === 4) ? 'highlight' : '';
	items.push( '<li id="obj' + data[i].index + '" class="' + className + '">' + data[i].name + '</li>' );
  });

  $( '<ul/>', {
	'class': 'person-list',
	html: items.join( '' )
  }).appendTo( '.list-container' );
  //console.clear();
  //console.log('----------------- CODE ENDS --------------------');
  console.clear();
  console.table(data, ['name','gender','email']);
});
```

### multiline console
- use `shift enter` to do multiline
```javascript
var a = 1;
var b = a;
var c = a + b;
```

### Snippets
- along `Sources` panel, there is `Snippets` panel
- add a snippets is to add a piece of codes
- right click to run it
- purpose of snippets: keep some often used but hard to remember codes blocks to copy and paste at hand

### Live edit
- in `Elements` panel, can select and edit it right away live
- in `Sources` panel, use `breakpoints`, `Watches` to monitor a variable changes
- go to console, and change the value of the variable you are monitoring
- move on to the next line of source code, and the changed value of the variable carries the new value onward
- you can delete or add lines of codes in the source file in `Sources` panel
- then right click and select `save` or `save as` to save the changed file to a new file or existing file in folder.

### Best practice
- do not minify code in development, because you won't be able to read it

### Call stack
- bottom up: the first ran code comes at bottom

### Clean up your codes
- create your own debug tools
- pre-debugging tools
	- "use strict": must use standard syntax

### other resources
https://youtu.be/-q1z8BPFItw?t=3634
