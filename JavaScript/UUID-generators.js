
/**
 * @description UUID generators
 * @link https://jsben.ch/xczxS
 * @returns {string}
 */

function b() {
	return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
		var r = Math.random()*16|0, v = c == 'x' ? r : (r&0x3|0x8);
		return v.toString(16);
	});
}
  
function e1() {
  var u='',i=0;
  while(i++<36) {
    var c='xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'[i-1],r=Math.random()*16|0,v=c=='x'?r:(r&0x3|0x8);
    u+=(c=='-'||c=='4')?c:v.toString(16)
  }
  return u;
}
  
function e2() {
  var u='',m='xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx',i=0,rb=Math.random()*0xffffffff|0;
  while(i++<36) {
    var c=m[i-1],r=rb&0xf,v=c=='x'?r:(r&0x3|0x8);
    u+=(c=='-'||c=='4')?c:v.toString(16);rb=i%8==0?Math.random()*0xffffffff|0:rb>>4
  }
  return u;
}
  
function e3() {
  var l='0123456789abcdef';
  var m='xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx';
  var u='',i=0,rb=Math.random()*0xffffffff|0;
  while(i++<36) {
    var c=m[i-1],r=rb&0xf,v=c=='x'?r:(r&0x3|0x8);
    u+=(c=='-'||c=='4')?c:l[v];rb=i%8==0?Math.random()*0xffffffff|0:rb>>4
  }
  return u;
}
  
function e4() {
  var h=['0','1','2','3','4','5','6','7','8','9','a','b','c','d','e','f'];
  var k=['x','x','x','x','x','x','x','x','-','x','x','x','x','-','4','x','x','x','-','y','x','x','x','-','x','x','x','x','x','x','x','x','x','x','x','x'];
  var u='',i=0,rb=Math.random()*0xffffffff|0;
  while(i++<36) {
    var c=k[i-1],r=rb&0xf,v=c=='x'?r:(r&0x3|0x8);
    u+=(c=='-'||c=='4')?c:h[v];rb=i%8==0?Math.random()*0xffffffff|0:rb>>4
  }
  return u;
}

var lut = []; for (var i=0; i<256; i++) { lut[i] = (i<16?'0':'')+(i).toString(16); }
function e5() {
  var k=['x','x','x','x','-','x','x','-','4','x','-','y','x','-','x','x','x','x','x','x'];
  var u='',i=0,rb=Math.random()*0xffffffff|0;
  while(i++<20) {
    var c=k[i-1],r=rb&0xff,v=c=='x'?r:(c=='y'?(r&0x3f|0x80):(r&0xf|0x40));
    u+=(c=='-')?c:lut[v];rb=i%4==0?Math.random()*0xffffffff|0:rb>>8
  }
  return u;
}
  
function e6() {
  var k=['x','x','-','x','-','4','-','y','-','x','x','x'];
  var u='',i=0,rb=Math.random()*0xffffffff|0;
  while(i++<12) {
    var c=k[i-1],r=rb&0xffff,v=c=='x'?r:(c=='y'?(r&0x3fff|0x8000):(r&0xfff|0x4000));
    u+=(c=='-')?c:(lut[v>>8]+lut[v&0xff]);rb=i&1?rb>>16:Math.random()*0xffffffff|0
  }
  return u;
}
  
function e7()
{
  var d0 = Math.random()*0xffffffff|0;
  var d1 = Math.random()*0xffffffff|0;
  var d2 = Math.random()*0xffffffff|0;
  var d3 = Math.random()*0xffffffff|0;
  return lut[d0&0xff]+lut[d0>>8&0xff]+lut[d0>>16&0xff]+lut[d0>>24&0xff]+'-'+
    lut[d1&0xff]+lut[d1>>8&0xff]+'-'+lut[d1>>16&0x0f|0x40]+lut[d1>>24&0xff]+'-'+
    lut[d2&0x3f|0x80]+lut[d2>>8&0xff]+'-'+lut[d2>>16&0xff]+lut[d2>>24&0xff]+
    lut[d3&0xff]+lut[d3>>8&0xff]+lut[d3>>16&0xff]+lut[d3>>24&0xff];
}
  
function s4() {
  return Math.floor((1 + Math.random()) * 0x10000)
    .toString(16)
    .substring(1);
}
  
function guid() {
  return s4() + s4() + '-' + s4() + '-' + s4() + '-' +
    s4() + '-' + s4() + s4() + s4();
}
  
function generateQuickGuid() {
  return Math.random().toString(36).substring(2, 15) + Math.random().toString(36).substring(2, 15);
}