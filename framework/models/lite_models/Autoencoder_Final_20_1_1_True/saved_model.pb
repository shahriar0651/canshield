??
??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
?
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

.
Identity

input"T
output"T"	
Ttype
\
	LeakyRelu
features"T
activations"T"
alphafloat%??L>"
Ttype0:
2
?
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
?
Mul
x"T
y"T
z"T"
Ttype:
2	?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
_
Pad

input"T
paddings"	Tpaddings
output"T"	
Ttype"
	Tpaddingstype0:
2	
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
?
ResizeNearestNeighbor
images"T
size
resized_images"T"
Ttype:
2
	"
align_cornersbool( "
half_pixel_centersbool( 
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.5.0-dev202103242v1.12.1-53598-gd0598853f858??
?
conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_nameconv2d_1/kernel
{
#conv2d_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_1/kernel*&
_output_shapes
: *
dtype0
r
conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_1/bias
k
!conv2d_1/bias/Read/ReadVariableOpReadVariableOpconv2d_1/bias*
_output_shapes
: *
dtype0
?
conv2d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_nameconv2d_2/kernel
{
#conv2d_2/kernel/Read/ReadVariableOpReadVariableOpconv2d_2/kernel*&
_output_shapes
: *
dtype0
r
conv2d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_2/bias
k
!conv2d_2/bias/Read/ReadVariableOpReadVariableOpconv2d_2/bias*
_output_shapes
:*
dtype0
?
conv2d_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_3/kernel
{
#conv2d_3/kernel/Read/ReadVariableOpReadVariableOpconv2d_3/kernel*&
_output_shapes
:*
dtype0
r
conv2d_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_3/bias
k
!conv2d_3/bias/Read/ReadVariableOpReadVariableOpconv2d_3/bias*
_output_shapes
:*
dtype0
?
conv2d_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_4/kernel
{
#conv2d_4/kernel/Read/ReadVariableOpReadVariableOpconv2d_4/kernel*&
_output_shapes
:*
dtype0
r
conv2d_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_4/bias
k
!conv2d_4/bias/Read/ReadVariableOpReadVariableOpconv2d_4/bias*
_output_shapes
:*
dtype0
?
conv2d_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_5/kernel
{
#conv2d_5/kernel/Read/ReadVariableOpReadVariableOpconv2d_5/kernel*&
_output_shapes
:*
dtype0
r
conv2d_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_5/bias
k
!conv2d_5/bias/Read/ReadVariableOpReadVariableOpconv2d_5/bias*
_output_shapes
:*
dtype0
?
conv2d_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_nameconv2d_6/kernel
{
#conv2d_6/kernel/Read/ReadVariableOpReadVariableOpconv2d_6/kernel*&
_output_shapes
: *
dtype0
r
conv2d_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_6/bias
k
!conv2d_6/bias/Read/ReadVariableOpReadVariableOpconv2d_6/bias*
_output_shapes
: *
dtype0
?
conv2d_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_nameconv2d_7/kernel
{
#conv2d_7/kernel/Read/ReadVariableOpReadVariableOpconv2d_7/kernel*&
_output_shapes
: *
dtype0
r
conv2d_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_7/bias
k
!conv2d_7/bias/Read/ReadVariableOpReadVariableOpconv2d_7/bias*
_output_shapes
:*
dtype0
\
iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_nameiter
U
iter/Read/ReadVariableOpReadVariableOpiter*
_output_shapes
: *
dtype0	
`
beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namebeta_1
Y
beta_1/Read/ReadVariableOpReadVariableOpbeta_1*
_output_shapes
: *
dtype0
`
beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namebeta_2
Y
beta_2/Read/ReadVariableOpReadVariableOpbeta_2*
_output_shapes
: *
dtype0
^
decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedecay
W
decay/Read/ReadVariableOpReadVariableOpdecay*
_output_shapes
: *
dtype0
n
learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
?
conv2d_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nameconv2d_1/kernel/m

%conv2d_1/kernel/m/Read/ReadVariableOpReadVariableOpconv2d_1/kernel/m*&
_output_shapes
: *
dtype0
v
conv2d_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_nameconv2d_1/bias/m
o
#conv2d_1/bias/m/Read/ReadVariableOpReadVariableOpconv2d_1/bias/m*
_output_shapes
: *
dtype0
?
conv2d_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nameconv2d_2/kernel/m

%conv2d_2/kernel/m/Read/ReadVariableOpReadVariableOpconv2d_2/kernel/m*&
_output_shapes
: *
dtype0
v
conv2d_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_2/bias/m
o
#conv2d_2/bias/m/Read/ReadVariableOpReadVariableOpconv2d_2/bias/m*
_output_shapes
:*
dtype0
?
conv2d_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameconv2d_3/kernel/m

%conv2d_3/kernel/m/Read/ReadVariableOpReadVariableOpconv2d_3/kernel/m*&
_output_shapes
:*
dtype0
v
conv2d_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_3/bias/m
o
#conv2d_3/bias/m/Read/ReadVariableOpReadVariableOpconv2d_3/bias/m*
_output_shapes
:*
dtype0
?
conv2d_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameconv2d_4/kernel/m

%conv2d_4/kernel/m/Read/ReadVariableOpReadVariableOpconv2d_4/kernel/m*&
_output_shapes
:*
dtype0
v
conv2d_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_4/bias/m
o
#conv2d_4/bias/m/Read/ReadVariableOpReadVariableOpconv2d_4/bias/m*
_output_shapes
:*
dtype0
?
conv2d_5/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameconv2d_5/kernel/m

%conv2d_5/kernel/m/Read/ReadVariableOpReadVariableOpconv2d_5/kernel/m*&
_output_shapes
:*
dtype0
v
conv2d_5/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_5/bias/m
o
#conv2d_5/bias/m/Read/ReadVariableOpReadVariableOpconv2d_5/bias/m*
_output_shapes
:*
dtype0
?
conv2d_6/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nameconv2d_6/kernel/m

%conv2d_6/kernel/m/Read/ReadVariableOpReadVariableOpconv2d_6/kernel/m*&
_output_shapes
: *
dtype0
v
conv2d_6/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_nameconv2d_6/bias/m
o
#conv2d_6/bias/m/Read/ReadVariableOpReadVariableOpconv2d_6/bias/m*
_output_shapes
: *
dtype0
?
conv2d_7/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nameconv2d_7/kernel/m

%conv2d_7/kernel/m/Read/ReadVariableOpReadVariableOpconv2d_7/kernel/m*&
_output_shapes
: *
dtype0
v
conv2d_7/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_7/bias/m
o
#conv2d_7/bias/m/Read/ReadVariableOpReadVariableOpconv2d_7/bias/m*
_output_shapes
:*
dtype0
?
conv2d_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nameconv2d_1/kernel/v

%conv2d_1/kernel/v/Read/ReadVariableOpReadVariableOpconv2d_1/kernel/v*&
_output_shapes
: *
dtype0
v
conv2d_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_nameconv2d_1/bias/v
o
#conv2d_1/bias/v/Read/ReadVariableOpReadVariableOpconv2d_1/bias/v*
_output_shapes
: *
dtype0
?
conv2d_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nameconv2d_2/kernel/v

%conv2d_2/kernel/v/Read/ReadVariableOpReadVariableOpconv2d_2/kernel/v*&
_output_shapes
: *
dtype0
v
conv2d_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_2/bias/v
o
#conv2d_2/bias/v/Read/ReadVariableOpReadVariableOpconv2d_2/bias/v*
_output_shapes
:*
dtype0
?
conv2d_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameconv2d_3/kernel/v

%conv2d_3/kernel/v/Read/ReadVariableOpReadVariableOpconv2d_3/kernel/v*&
_output_shapes
:*
dtype0
v
conv2d_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_3/bias/v
o
#conv2d_3/bias/v/Read/ReadVariableOpReadVariableOpconv2d_3/bias/v*
_output_shapes
:*
dtype0
?
conv2d_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameconv2d_4/kernel/v

%conv2d_4/kernel/v/Read/ReadVariableOpReadVariableOpconv2d_4/kernel/v*&
_output_shapes
:*
dtype0
v
conv2d_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_4/bias/v
o
#conv2d_4/bias/v/Read/ReadVariableOpReadVariableOpconv2d_4/bias/v*
_output_shapes
:*
dtype0
?
conv2d_5/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameconv2d_5/kernel/v

%conv2d_5/kernel/v/Read/ReadVariableOpReadVariableOpconv2d_5/kernel/v*&
_output_shapes
:*
dtype0
v
conv2d_5/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_5/bias/v
o
#conv2d_5/bias/v/Read/ReadVariableOpReadVariableOpconv2d_5/bias/v*
_output_shapes
:*
dtype0
?
conv2d_6/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nameconv2d_6/kernel/v

%conv2d_6/kernel/v/Read/ReadVariableOpReadVariableOpconv2d_6/kernel/v*&
_output_shapes
: *
dtype0
v
conv2d_6/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_nameconv2d_6/bias/v
o
#conv2d_6/bias/v/Read/ReadVariableOpReadVariableOpconv2d_6/bias/v*
_output_shapes
: *
dtype0
?
conv2d_7/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nameconv2d_7/kernel/v

%conv2d_7/kernel/v/Read/ReadVariableOpReadVariableOpconv2d_7/kernel/v*&
_output_shapes
: *
dtype0
v
conv2d_7/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_7/bias/v
o
#conv2d_7/bias/v/Read/ReadVariableOpReadVariableOpconv2d_7/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
?h
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?h
value?hB?h B?h
?
layer-0
layer_with_weights-0
layer-1
layer-2
layer-3
layer_with_weights-1
layer-4
layer-5
layer-6
layer_with_weights-2
layer-7
	layer-8

layer-9
layer_with_weights-3
layer-10
layer-11
layer-12
layer_with_weights-4
layer-13
layer-14
layer-15
layer_with_weights-5
layer-16
layer-17
layer-18
layer_with_weights-6
layer-19
layer-20
	optimizer
regularization_losses
	variables
trainable_variables
	keras_api

signatures
R
regularization_losses
	variables
trainable_variables
	keras_api
h

 kernel
!bias
"regularization_losses
#	variables
$trainable_variables
%	keras_api
R
&regularization_losses
'	variables
(trainable_variables
)	keras_api
R
*regularization_losses
+	variables
,trainable_variables
-	keras_api
h

.kernel
/bias
0regularization_losses
1	variables
2trainable_variables
3	keras_api
R
4regularization_losses
5	variables
6trainable_variables
7	keras_api
R
8regularization_losses
9	variables
:trainable_variables
;	keras_api
h

<kernel
=bias
>regularization_losses
?	variables
@trainable_variables
A	keras_api
R
Bregularization_losses
C	variables
Dtrainable_variables
E	keras_api
R
Fregularization_losses
G	variables
Htrainable_variables
I	keras_api
h

Jkernel
Kbias
Lregularization_losses
M	variables
Ntrainable_variables
O	keras_api
R
Pregularization_losses
Q	variables
Rtrainable_variables
S	keras_api
R
Tregularization_losses
U	variables
Vtrainable_variables
W	keras_api
h

Xkernel
Ybias
Zregularization_losses
[	variables
\trainable_variables
]	keras_api
R
^regularization_losses
_	variables
`trainable_variables
a	keras_api
R
bregularization_losses
c	variables
dtrainable_variables
e	keras_api
h

fkernel
gbias
hregularization_losses
i	variables
jtrainable_variables
k	keras_api
R
lregularization_losses
m	variables
ntrainable_variables
o	keras_api
R
pregularization_losses
q	variables
rtrainable_variables
s	keras_api
h

tkernel
ubias
vregularization_losses
w	variables
xtrainable_variables
y	keras_api
R
zregularization_losses
{	variables
|trainable_variables
}	keras_api
?
~iter

beta_1
?beta_2

?decay
?learning_rate m?!m?.m?/m?<m?=m?Jm?Km?Xm?Ym?fm?gm?tm?um? v?!v?.v?/v?<v?=v?Jv?Kv?Xv?Yv?fv?gv?tv?uv?
 
f
 0
!1
.2
/3
<4
=5
J6
K7
X8
Y9
f10
g11
t12
u13
f
 0
!1
.2
/3
<4
=5
J6
K7
X8
Y9
f10
g11
t12
u13
?
?layers
?non_trainable_variables
regularization_losses
	variables
 ?layer_regularization_losses
?metrics
?layer_metrics
trainable_variables
 
 
 
 
?
?layers
?non_trainable_variables
regularization_losses
	variables
 ?layer_regularization_losses
?metrics
?layer_metrics
trainable_variables
[Y
VARIABLE_VALUEconv2d_1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

 0
!1

 0
!1
?
?layers
?non_trainable_variables
"regularization_losses
#	variables
 ?layer_regularization_losses
?metrics
?layer_metrics
$trainable_variables
 
 
 
?
?layers
?non_trainable_variables
&regularization_losses
'	variables
 ?layer_regularization_losses
?metrics
?layer_metrics
(trainable_variables
 
 
 
?
?layers
?non_trainable_variables
*regularization_losses
+	variables
 ?layer_regularization_losses
?metrics
?layer_metrics
,trainable_variables
[Y
VARIABLE_VALUEconv2d_2/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_2/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

.0
/1

.0
/1
?
?layers
?non_trainable_variables
0regularization_losses
1	variables
 ?layer_regularization_losses
?metrics
?layer_metrics
2trainable_variables
 
 
 
?
?layers
?non_trainable_variables
4regularization_losses
5	variables
 ?layer_regularization_losses
?metrics
?layer_metrics
6trainable_variables
 
 
 
?
?layers
?non_trainable_variables
8regularization_losses
9	variables
 ?layer_regularization_losses
?metrics
?layer_metrics
:trainable_variables
[Y
VARIABLE_VALUEconv2d_3/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_3/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

<0
=1

<0
=1
?
?layers
?non_trainable_variables
>regularization_losses
?	variables
 ?layer_regularization_losses
?metrics
?layer_metrics
@trainable_variables
 
 
 
?
?layers
?non_trainable_variables
Bregularization_losses
C	variables
 ?layer_regularization_losses
?metrics
?layer_metrics
Dtrainable_variables
 
 
 
?
?layers
?non_trainable_variables
Fregularization_losses
G	variables
 ?layer_regularization_losses
?metrics
?layer_metrics
Htrainable_variables
[Y
VARIABLE_VALUEconv2d_4/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_4/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 

J0
K1

J0
K1
?
?layers
?non_trainable_variables
Lregularization_losses
M	variables
 ?layer_regularization_losses
?metrics
?layer_metrics
Ntrainable_variables
 
 
 
?
?layers
?non_trainable_variables
Pregularization_losses
Q	variables
 ?layer_regularization_losses
?metrics
?layer_metrics
Rtrainable_variables
 
 
 
?
?layers
?non_trainable_variables
Tregularization_losses
U	variables
 ?layer_regularization_losses
?metrics
?layer_metrics
Vtrainable_variables
[Y
VARIABLE_VALUEconv2d_5/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_5/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
 

X0
Y1

X0
Y1
?
?layers
?non_trainable_variables
Zregularization_losses
[	variables
 ?layer_regularization_losses
?metrics
?layer_metrics
\trainable_variables
 
 
 
?
?layers
?non_trainable_variables
^regularization_losses
_	variables
 ?layer_regularization_losses
?metrics
?layer_metrics
`trainable_variables
 
 
 
?
?layers
?non_trainable_variables
bregularization_losses
c	variables
 ?layer_regularization_losses
?metrics
?layer_metrics
dtrainable_variables
[Y
VARIABLE_VALUEconv2d_6/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_6/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE
 

f0
g1

f0
g1
?
?layers
?non_trainable_variables
hregularization_losses
i	variables
 ?layer_regularization_losses
?metrics
?layer_metrics
jtrainable_variables
 
 
 
?
?layers
?non_trainable_variables
lregularization_losses
m	variables
 ?layer_regularization_losses
?metrics
?layer_metrics
ntrainable_variables
 
 
 
?
?layers
?non_trainable_variables
pregularization_losses
q	variables
 ?layer_regularization_losses
?metrics
?layer_metrics
rtrainable_variables
[Y
VARIABLE_VALUEconv2d_7/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_7/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE
 

t0
u1

t0
u1
?
?layers
?non_trainable_variables
vregularization_losses
w	variables
 ?layer_regularization_losses
?metrics
?layer_metrics
xtrainable_variables
 
 
 
?
?layers
?non_trainable_variables
zregularization_losses
{	variables
 ?layer_regularization_losses
?metrics
?layer_metrics
|trainable_variables
CA
VARIABLE_VALUEiter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
GE
VARIABLE_VALUEbeta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
GE
VARIABLE_VALUEbeta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
EC
VARIABLE_VALUEdecay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUElearning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
?
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
 
 

?0
?1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
8

?total

?count
?	variables
?	keras_api
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
yw
VARIABLE_VALUEconv2d_1/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEconv2d_1/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEconv2d_2/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEconv2d_2/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEconv2d_3/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEconv2d_3/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEconv2d_4/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEconv2d_4/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEconv2d_5/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEconv2d_5/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEconv2d_6/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEconv2d_6/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEconv2d_7/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEconv2d_7/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEconv2d_1/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEconv2d_1/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEconv2d_2/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEconv2d_2/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEconv2d_3/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEconv2d_3/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEconv2d_4/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEconv2d_4/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEconv2d_5/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEconv2d_5/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEconv2d_6/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEconv2d_6/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEconv2d_7/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEconv2d_7/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
&serving_default_zero_padding2d_1_inputPlaceholder*/
_output_shapes
:?????????*
dtype0*$
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCall&serving_default_zero_padding2d_1_inputconv2d_1/kernelconv2d_1/biasconv2d_2/kernelconv2d_2/biasconv2d_3/kernelconv2d_3/biasconv2d_4/kernelconv2d_4/biasconv2d_5/kernelconv2d_5/biasconv2d_6/kernelconv2d_6/biasconv2d_7/kernelconv2d_7/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *,
f'R%
#__inference_signature_wrapper_26595
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#conv2d_1/kernel/Read/ReadVariableOp!conv2d_1/bias/Read/ReadVariableOp#conv2d_2/kernel/Read/ReadVariableOp!conv2d_2/bias/Read/ReadVariableOp#conv2d_3/kernel/Read/ReadVariableOp!conv2d_3/bias/Read/ReadVariableOp#conv2d_4/kernel/Read/ReadVariableOp!conv2d_4/bias/Read/ReadVariableOp#conv2d_5/kernel/Read/ReadVariableOp!conv2d_5/bias/Read/ReadVariableOp#conv2d_6/kernel/Read/ReadVariableOp!conv2d_6/bias/Read/ReadVariableOp#conv2d_7/kernel/Read/ReadVariableOp!conv2d_7/bias/Read/ReadVariableOpiter/Read/ReadVariableOpbeta_1/Read/ReadVariableOpbeta_2/Read/ReadVariableOpdecay/Read/ReadVariableOp!learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp%conv2d_1/kernel/m/Read/ReadVariableOp#conv2d_1/bias/m/Read/ReadVariableOp%conv2d_2/kernel/m/Read/ReadVariableOp#conv2d_2/bias/m/Read/ReadVariableOp%conv2d_3/kernel/m/Read/ReadVariableOp#conv2d_3/bias/m/Read/ReadVariableOp%conv2d_4/kernel/m/Read/ReadVariableOp#conv2d_4/bias/m/Read/ReadVariableOp%conv2d_5/kernel/m/Read/ReadVariableOp#conv2d_5/bias/m/Read/ReadVariableOp%conv2d_6/kernel/m/Read/ReadVariableOp#conv2d_6/bias/m/Read/ReadVariableOp%conv2d_7/kernel/m/Read/ReadVariableOp#conv2d_7/bias/m/Read/ReadVariableOp%conv2d_1/kernel/v/Read/ReadVariableOp#conv2d_1/bias/v/Read/ReadVariableOp%conv2d_2/kernel/v/Read/ReadVariableOp#conv2d_2/bias/v/Read/ReadVariableOp%conv2d_3/kernel/v/Read/ReadVariableOp#conv2d_3/bias/v/Read/ReadVariableOp%conv2d_4/kernel/v/Read/ReadVariableOp#conv2d_4/bias/v/Read/ReadVariableOp%conv2d_5/kernel/v/Read/ReadVariableOp#conv2d_5/bias/v/Read/ReadVariableOp%conv2d_6/kernel/v/Read/ReadVariableOp#conv2d_6/bias/v/Read/ReadVariableOp%conv2d_7/kernel/v/Read/ReadVariableOp#conv2d_7/bias/v/Read/ReadVariableOpConst*@
Tin9
725	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *'
f"R 
__inference__traced_save_27020
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_1/kernelconv2d_1/biasconv2d_2/kernelconv2d_2/biasconv2d_3/kernelconv2d_3/biasconv2d_4/kernelconv2d_4/biasconv2d_5/kernelconv2d_5/biasconv2d_6/kernelconv2d_6/biasconv2d_7/kernelconv2d_7/biasiterbeta_1beta_2decaylearning_ratetotalcounttotal_1count_1conv2d_1/kernel/mconv2d_1/bias/mconv2d_2/kernel/mconv2d_2/bias/mconv2d_3/kernel/mconv2d_3/bias/mconv2d_4/kernel/mconv2d_4/bias/mconv2d_5/kernel/mconv2d_5/bias/mconv2d_6/kernel/mconv2d_6/bias/mconv2d_7/kernel/mconv2d_7/bias/mconv2d_1/kernel/vconv2d_1/bias/vconv2d_2/kernel/vconv2d_2/bias/vconv2d_3/kernel/vconv2d_3/bias/vconv2d_4/kernel/vconv2d_4/bias/vconv2d_5/kernel/vconv2d_5/bias/vconv2d_6/kernel/vconv2d_6/bias/vconv2d_7/kernel/vconv2d_7/bias/v*?
Tin8
624*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? **
f%R#
!__inference__traced_restore_27183??

?
K
/__inference_max_pooling2d_1_layer_call_fn_26064

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_260582
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
C__inference_conv2d_6_layer_call_and_return_conditional_losses_26295

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+??????????????????????????? *
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
#__inference_signature_wrapper_26595
zero_padding2d_1_input!
unknown: 
	unknown_0: #
	unknown_1: 
	unknown_2:#
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:#
	unknown_7:
	unknown_8:#
	unknown_9: 

unknown_10: $

unknown_11: 

unknown_12:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallzero_padding2d_1_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *)
f$R"
 __inference__wrapped_model_260392
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:?????????: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:g c
/
_output_shapes
:?????????
0
_user_specified_namezero_padding2d_1_input
?
?
C__inference_conv2d_7_layer_call_and_return_conditional_losses_26844

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????2	
BiasAdd{
SigmoidSigmoidBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2	
Sigmoid?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+??????????????????????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?

?
C__inference_conv2d_4_layer_call_and_return_conditional_losses_26247

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?T
?
G__inference_sequential_1_layer_call_and_return_conditional_losses_26523

inputs(
conv2d_1_26474: 
conv2d_1_26476: (
conv2d_2_26481: 
conv2d_2_26483:(
conv2d_3_26488:
conv2d_3_26490:(
conv2d_4_26495:
conv2d_4_26497:(
conv2d_5_26502:
conv2d_5_26504:(
conv2d_6_26509: 
conv2d_6_26511: (
conv2d_7_26516: 
conv2d_7_26518:
identity?? conv2d_1/StatefulPartitionedCall? conv2d_2/StatefulPartitionedCall? conv2d_3/StatefulPartitionedCall? conv2d_4/StatefulPartitionedCall? conv2d_5/StatefulPartitionedCall? conv2d_6/StatefulPartitionedCall? conv2d_7/StatefulPartitionedCall?
 zero_padding2d_1/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_zero_padding2d_1_layer_call_and_return_conditional_losses_260462"
 zero_padding2d_1/PartitionedCall?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall)zero_padding2d_1/PartitionedCall:output:0conv2d_1_26474conv2d_1_26476*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_261752"
 conv2d_1/StatefulPartitionedCall?
leaky_re_lu_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_261862
leaky_re_lu_1/PartitionedCall?
max_pooling2d_1/PartitionedCallPartitionedCall&leaky_re_lu_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_260582!
max_pooling2d_1/PartitionedCall?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0conv2d_2_26481conv2d_2_26483*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_2_layer_call_and_return_conditional_losses_261992"
 conv2d_2/StatefulPartitionedCall?
leaky_re_lu_2/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_262102
leaky_re_lu_2/PartitionedCall?
max_pooling2d_2/PartitionedCallPartitionedCall&leaky_re_lu_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_260702!
max_pooling2d_2/PartitionedCall?
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_2/PartitionedCall:output:0conv2d_3_26488conv2d_3_26490*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_3_layer_call_and_return_conditional_losses_262232"
 conv2d_3/StatefulPartitionedCall?
leaky_re_lu_3/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_leaky_re_lu_3_layer_call_and_return_conditional_losses_262342
leaky_re_lu_3/PartitionedCall?
max_pooling2d_3/PartitionedCallPartitionedCall&leaky_re_lu_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_260822!
max_pooling2d_3/PartitionedCall?
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_3/PartitionedCall:output:0conv2d_4_26495conv2d_4_26497*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_4_layer_call_and_return_conditional_losses_262472"
 conv2d_4/StatefulPartitionedCall?
leaky_re_lu_4/PartitionedCallPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_leaky_re_lu_4_layer_call_and_return_conditional_losses_262582
leaky_re_lu_4/PartitionedCall?
up_sampling2d_1/PartitionedCallPartitionedCall&leaky_re_lu_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_261012!
up_sampling2d_1/PartitionedCall?
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_1/PartitionedCall:output:0conv2d_5_26502conv2d_5_26504*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_5_layer_call_and_return_conditional_losses_262712"
 conv2d_5/StatefulPartitionedCall?
leaky_re_lu_5/PartitionedCallPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_leaky_re_lu_5_layer_call_and_return_conditional_losses_262822
leaky_re_lu_5/PartitionedCall?
up_sampling2d_2/PartitionedCallPartitionedCall&leaky_re_lu_5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_up_sampling2d_2_layer_call_and_return_conditional_losses_261202!
up_sampling2d_2/PartitionedCall?
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_2/PartitionedCall:output:0conv2d_6_26509conv2d_6_26511*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_6_layer_call_and_return_conditional_losses_262952"
 conv2d_6/StatefulPartitionedCall?
leaky_re_lu_6/PartitionedCallPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_leaky_re_lu_6_layer_call_and_return_conditional_losses_263062
leaky_re_lu_6/PartitionedCall?
up_sampling2d_3/PartitionedCallPartitionedCall&leaky_re_lu_6/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_up_sampling2d_3_layer_call_and_return_conditional_losses_261392!
up_sampling2d_3/PartitionedCall?
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_3/PartitionedCall:output:0conv2d_7_26516conv2d_7_26518*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_7_layer_call_and_return_conditional_losses_263202"
 conv2d_7/StatefulPartitionedCall?
cropping2d_1/PartitionedCallPartitionedCall)conv2d_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_cropping2d_1_layer_call_and_return_conditional_losses_261542
cropping2d_1/PartitionedCall?
IdentityIdentity%cropping2d_1/PartitionedCall:output:0!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall!^conv2d_7/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:?????????: : : : : : : : : : : : : : 2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
C__inference_conv2d_2_layer_call_and_return_conditional_losses_26199

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
K
/__inference_max_pooling2d_2_layer_call_fn_26076

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_260702
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
f
J__inference_up_sampling2d_2_layer_call_and_return_conditional_losses_26120

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
strided_slice_
ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
Const^
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:2
mul?
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4????????????????????????????????????*
half_pixel_centers(2
resize/ResizeNearestNeighbor?
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?h
?
G__inference_sequential_1_layer_call_and_return_conditional_losses_26669

inputsA
'conv2d_1_conv2d_readvariableop_resource: 6
(conv2d_1_biasadd_readvariableop_resource: A
'conv2d_2_conv2d_readvariableop_resource: 6
(conv2d_2_biasadd_readvariableop_resource:A
'conv2d_3_conv2d_readvariableop_resource:6
(conv2d_3_biasadd_readvariableop_resource:A
'conv2d_4_conv2d_readvariableop_resource:6
(conv2d_4_biasadd_readvariableop_resource:A
'conv2d_5_conv2d_readvariableop_resource:6
(conv2d_5_biasadd_readvariableop_resource:A
'conv2d_6_conv2d_readvariableop_resource: 6
(conv2d_6_biasadd_readvariableop_resource: A
'conv2d_7_conv2d_readvariableop_resource: 6
(conv2d_7_biasadd_readvariableop_resource:
identity??conv2d_1/BiasAdd/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOp?conv2d_2/BiasAdd/ReadVariableOp?conv2d_2/Conv2D/ReadVariableOp?conv2d_3/BiasAdd/ReadVariableOp?conv2d_3/Conv2D/ReadVariableOp?conv2d_4/BiasAdd/ReadVariableOp?conv2d_4/Conv2D/ReadVariableOp?conv2d_5/BiasAdd/ReadVariableOp?conv2d_5/Conv2D/ReadVariableOp?conv2d_6/BiasAdd/ReadVariableOp?conv2d_6/Conv2D/ReadVariableOp?conv2d_7/BiasAdd/ReadVariableOp?conv2d_7/Conv2D/ReadVariableOp?
zero_padding2d_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                             2
zero_padding2d_1/Pad/paddings?
zero_padding2d_1/PadPadinputs&zero_padding2d_1/Pad/paddings:output:0*
T0*/
_output_shapes
:?????????2
zero_padding2d_1/Pad?
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02 
conv2d_1/Conv2D/ReadVariableOp?
conv2d_1/Conv2DConv2Dzero_padding2d_1/Pad:output:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2
conv2d_1/Conv2D?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_1/BiasAdd/ReadVariableOp?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
conv2d_1/BiasAdd?
leaky_re_lu_1/LeakyRelu	LeakyReluconv2d_1/BiasAdd:output:0*/
_output_shapes
:????????? 2
leaky_re_lu_1/LeakyRelu?
max_pooling2d_1/MaxPoolMaxPool%leaky_re_lu_1/LeakyRelu:activations:0*/
_output_shapes
:????????? *
ksize
*
paddingSAME*
strides
2
max_pooling2d_1/MaxPool?
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02 
conv2d_2/Conv2D/ReadVariableOp?
conv2d_2/Conv2DConv2D max_pooling2d_1/MaxPool:output:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
conv2d_2/Conv2D?
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_2/BiasAdd/ReadVariableOp?
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
conv2d_2/BiasAdd?
leaky_re_lu_2/LeakyRelu	LeakyReluconv2d_2/BiasAdd:output:0*/
_output_shapes
:?????????2
leaky_re_lu_2/LeakyRelu?
max_pooling2d_2/MaxPoolMaxPool%leaky_re_lu_2/LeakyRelu:activations:0*/
_output_shapes
:?????????*
ksize
*
paddingSAME*
strides
2
max_pooling2d_2/MaxPool?
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02 
conv2d_3/Conv2D/ReadVariableOp?
conv2d_3/Conv2DConv2D max_pooling2d_2/MaxPool:output:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
conv2d_3/Conv2D?
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_3/BiasAdd/ReadVariableOp?
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
conv2d_3/BiasAdd?
leaky_re_lu_3/LeakyRelu	LeakyReluconv2d_3/BiasAdd:output:0*/
_output_shapes
:?????????2
leaky_re_lu_3/LeakyRelu?
max_pooling2d_3/MaxPoolMaxPool%leaky_re_lu_3/LeakyRelu:activations:0*/
_output_shapes
:?????????*
ksize
*
paddingSAME*
strides
2
max_pooling2d_3/MaxPool?
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02 
conv2d_4/Conv2D/ReadVariableOp?
conv2d_4/Conv2DConv2D max_pooling2d_3/MaxPool:output:0&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
conv2d_4/Conv2D?
conv2d_4/BiasAdd/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_4/BiasAdd/ReadVariableOp?
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0'conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
conv2d_4/BiasAdd?
leaky_re_lu_4/LeakyRelu	LeakyReluconv2d_4/BiasAdd:output:0*/
_output_shapes
:?????????2
leaky_re_lu_4/LeakyRelu
up_sampling2d_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_1/Const?
up_sampling2d_1/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_1/Const_1?
up_sampling2d_1/mulMulup_sampling2d_1/Const:output:0 up_sampling2d_1/Const_1:output:0*
T0*
_output_shapes
:2
up_sampling2d_1/mul?
,up_sampling2d_1/resize/ResizeNearestNeighborResizeNearestNeighbor%leaky_re_lu_4/LeakyRelu:activations:0up_sampling2d_1/mul:z:0*
T0*/
_output_shapes
:?????????*
half_pixel_centers(2.
,up_sampling2d_1/resize/ResizeNearestNeighbor?
conv2d_5/Conv2D/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02 
conv2d_5/Conv2D/ReadVariableOp?
conv2d_5/Conv2DConv2D=up_sampling2d_1/resize/ResizeNearestNeighbor:resized_images:0&conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
conv2d_5/Conv2D?
conv2d_5/BiasAdd/ReadVariableOpReadVariableOp(conv2d_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_5/BiasAdd/ReadVariableOp?
conv2d_5/BiasAddBiasAddconv2d_5/Conv2D:output:0'conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
conv2d_5/BiasAdd?
leaky_re_lu_5/LeakyRelu	LeakyReluconv2d_5/BiasAdd:output:0*/
_output_shapes
:?????????2
leaky_re_lu_5/LeakyRelu
up_sampling2d_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_2/Const?
up_sampling2d_2/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_2/Const_1?
up_sampling2d_2/mulMulup_sampling2d_2/Const:output:0 up_sampling2d_2/Const_1:output:0*
T0*
_output_shapes
:2
up_sampling2d_2/mul?
,up_sampling2d_2/resize/ResizeNearestNeighborResizeNearestNeighbor%leaky_re_lu_5/LeakyRelu:activations:0up_sampling2d_2/mul:z:0*
T0*/
_output_shapes
:?????????*
half_pixel_centers(2.
,up_sampling2d_2/resize/ResizeNearestNeighbor?
conv2d_6/Conv2D/ReadVariableOpReadVariableOp'conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02 
conv2d_6/Conv2D/ReadVariableOp?
conv2d_6/Conv2DConv2D=up_sampling2d_2/resize/ResizeNearestNeighbor:resized_images:0&conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2
conv2d_6/Conv2D?
conv2d_6/BiasAdd/ReadVariableOpReadVariableOp(conv2d_6_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_6/BiasAdd/ReadVariableOp?
conv2d_6/BiasAddBiasAddconv2d_6/Conv2D:output:0'conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
conv2d_6/BiasAdd?
leaky_re_lu_6/LeakyRelu	LeakyReluconv2d_6/BiasAdd:output:0*/
_output_shapes
:????????? 2
leaky_re_lu_6/LeakyRelu
up_sampling2d_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_3/Const?
up_sampling2d_3/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_3/Const_1?
up_sampling2d_3/mulMulup_sampling2d_3/Const:output:0 up_sampling2d_3/Const_1:output:0*
T0*
_output_shapes
:2
up_sampling2d_3/mul?
,up_sampling2d_3/resize/ResizeNearestNeighborResizeNearestNeighbor%leaky_re_lu_6/LeakyRelu:activations:0up_sampling2d_3/mul:z:0*
T0*/
_output_shapes
:????????? *
half_pixel_centers(2.
,up_sampling2d_3/resize/ResizeNearestNeighbor?
conv2d_7/Conv2D/ReadVariableOpReadVariableOp'conv2d_7_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02 
conv2d_7/Conv2D/ReadVariableOp?
conv2d_7/Conv2DConv2D=up_sampling2d_3/resize/ResizeNearestNeighbor:resized_images:0&conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
conv2d_7/Conv2D?
conv2d_7/BiasAdd/ReadVariableOpReadVariableOp(conv2d_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_7/BiasAdd/ReadVariableOp?
conv2d_7/BiasAddBiasAddconv2d_7/Conv2D:output:0'conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
conv2d_7/BiasAdd?
conv2d_7/SigmoidSigmoidconv2d_7/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
conv2d_7/Sigmoid?
 cropping2d_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"              2"
 cropping2d_1/strided_slice/stack?
"cropping2d_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"    ????????    2$
"cropping2d_1/strided_slice/stack_1?
"cropping2d_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2$
"cropping2d_1/strided_slice/stack_2?
cropping2d_1/strided_sliceStridedSliceconv2d_7/Sigmoid:y:0)cropping2d_1/strided_slice/stack:output:0+cropping2d_1/strided_slice/stack_1:output:0+cropping2d_1/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:?????????*

begin_mask	*
end_mask	2
cropping2d_1/strided_slice?
IdentityIdentity#cropping2d_1/strided_slice:output:0 ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp ^conv2d_4/BiasAdd/ReadVariableOp^conv2d_4/Conv2D/ReadVariableOp ^conv2d_5/BiasAdd/ReadVariableOp^conv2d_5/Conv2D/ReadVariableOp ^conv2d_6/BiasAdd/ReadVariableOp^conv2d_6/Conv2D/ReadVariableOp ^conv2d_7/BiasAdd/ReadVariableOp^conv2d_7/Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:?????????: : : : : : : : : : : : : : 2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp2B
conv2d_4/BiasAdd/ReadVariableOpconv2d_4/BiasAdd/ReadVariableOp2@
conv2d_4/Conv2D/ReadVariableOpconv2d_4/Conv2D/ReadVariableOp2B
conv2d_5/BiasAdd/ReadVariableOpconv2d_5/BiasAdd/ReadVariableOp2@
conv2d_5/Conv2D/ReadVariableOpconv2d_5/Conv2D/ReadVariableOp2B
conv2d_6/BiasAdd/ReadVariableOpconv2d_6/BiasAdd/ReadVariableOp2@
conv2d_6/Conv2D/ReadVariableOpconv2d_6/Conv2D/ReadVariableOp2B
conv2d_7/BiasAdd/ReadVariableOpconv2d_7/BiasAdd/ReadVariableOp2@
conv2d_7/Conv2D/ReadVariableOpconv2d_7/Conv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
,__inference_sequential_1_layer_call_fn_26468
zero_padding2d_1_input!
unknown: 
	unknown_0: #
	unknown_1: 
	unknown_2:#
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:#
	unknown_7:
	unknown_8:#
	unknown_9: 

unknown_10: $

unknown_11: 

unknown_12:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallzero_padding2d_1_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_264372
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:?????????: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:g c
/
_output_shapes
:?????????
0
_user_specified_namezero_padding2d_1_input
?

?
C__inference_conv2d_1_layer_call_and_return_conditional_losses_26175

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
,__inference_sequential_1_layer_call_fn_26554
zero_padding2d_1_input!
unknown: 
	unknown_0: #
	unknown_1: 
	unknown_2:#
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:#
	unknown_7:
	unknown_8:#
	unknown_9: 

unknown_10: $

unknown_11: 

unknown_12:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallzero_padding2d_1_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_265232
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:?????????: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:g c
/
_output_shapes
:?????????
0
_user_specified_namezero_padding2d_1_input
?

?
C__inference_conv2d_4_layer_call_and_return_conditional_losses_26798

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?U
?
G__inference_sequential_1_layer_call_and_return_conditional_losses_26381
zero_padding2d_1_input(
conv2d_1_26332: 
conv2d_1_26334: (
conv2d_2_26339: 
conv2d_2_26341:(
conv2d_3_26346:
conv2d_3_26348:(
conv2d_4_26353:
conv2d_4_26355:(
conv2d_5_26360:
conv2d_5_26362:(
conv2d_6_26367: 
conv2d_6_26369: (
conv2d_7_26374: 
conv2d_7_26376:
identity?? conv2d_1/StatefulPartitionedCall? conv2d_2/StatefulPartitionedCall? conv2d_3/StatefulPartitionedCall? conv2d_4/StatefulPartitionedCall? conv2d_5/StatefulPartitionedCall? conv2d_6/StatefulPartitionedCall? conv2d_7/StatefulPartitionedCall?
 zero_padding2d_1/PartitionedCallPartitionedCallzero_padding2d_1_input*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_zero_padding2d_1_layer_call_and_return_conditional_losses_260462"
 zero_padding2d_1/PartitionedCall?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall)zero_padding2d_1/PartitionedCall:output:0conv2d_1_26332conv2d_1_26334*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_261752"
 conv2d_1/StatefulPartitionedCall?
leaky_re_lu_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_261862
leaky_re_lu_1/PartitionedCall?
max_pooling2d_1/PartitionedCallPartitionedCall&leaky_re_lu_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_260582!
max_pooling2d_1/PartitionedCall?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0conv2d_2_26339conv2d_2_26341*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_2_layer_call_and_return_conditional_losses_261992"
 conv2d_2/StatefulPartitionedCall?
leaky_re_lu_2/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_262102
leaky_re_lu_2/PartitionedCall?
max_pooling2d_2/PartitionedCallPartitionedCall&leaky_re_lu_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_260702!
max_pooling2d_2/PartitionedCall?
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_2/PartitionedCall:output:0conv2d_3_26346conv2d_3_26348*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_3_layer_call_and_return_conditional_losses_262232"
 conv2d_3/StatefulPartitionedCall?
leaky_re_lu_3/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_leaky_re_lu_3_layer_call_and_return_conditional_losses_262342
leaky_re_lu_3/PartitionedCall?
max_pooling2d_3/PartitionedCallPartitionedCall&leaky_re_lu_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_260822!
max_pooling2d_3/PartitionedCall?
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_3/PartitionedCall:output:0conv2d_4_26353conv2d_4_26355*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_4_layer_call_and_return_conditional_losses_262472"
 conv2d_4/StatefulPartitionedCall?
leaky_re_lu_4/PartitionedCallPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_leaky_re_lu_4_layer_call_and_return_conditional_losses_262582
leaky_re_lu_4/PartitionedCall?
up_sampling2d_1/PartitionedCallPartitionedCall&leaky_re_lu_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_261012!
up_sampling2d_1/PartitionedCall?
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_1/PartitionedCall:output:0conv2d_5_26360conv2d_5_26362*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_5_layer_call_and_return_conditional_losses_262712"
 conv2d_5/StatefulPartitionedCall?
leaky_re_lu_5/PartitionedCallPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_leaky_re_lu_5_layer_call_and_return_conditional_losses_262822
leaky_re_lu_5/PartitionedCall?
up_sampling2d_2/PartitionedCallPartitionedCall&leaky_re_lu_5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_up_sampling2d_2_layer_call_and_return_conditional_losses_261202!
up_sampling2d_2/PartitionedCall?
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_2/PartitionedCall:output:0conv2d_6_26367conv2d_6_26369*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_6_layer_call_and_return_conditional_losses_262952"
 conv2d_6/StatefulPartitionedCall?
leaky_re_lu_6/PartitionedCallPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_leaky_re_lu_6_layer_call_and_return_conditional_losses_263062
leaky_re_lu_6/PartitionedCall?
up_sampling2d_3/PartitionedCallPartitionedCall&leaky_re_lu_6/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_up_sampling2d_3_layer_call_and_return_conditional_losses_261392!
up_sampling2d_3/PartitionedCall?
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_3/PartitionedCall:output:0conv2d_7_26374conv2d_7_26376*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_7_layer_call_and_return_conditional_losses_263202"
 conv2d_7/StatefulPartitionedCall?
cropping2d_1/PartitionedCallPartitionedCall)conv2d_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_cropping2d_1_layer_call_and_return_conditional_losses_261542
cropping2d_1/PartitionedCall?
IdentityIdentity%cropping2d_1/PartitionedCall:output:0!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall!^conv2d_7/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:?????????: : : : : : : : : : : : : : 2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall:g c
/
_output_shapes
:?????????
0
_user_specified_namezero_padding2d_1_input
?
?
C__inference_conv2d_5_layer_call_and_return_conditional_losses_26271

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
L
0__inference_zero_padding2d_1_layer_call_fn_26052

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_zero_padding2d_1_layer_call_and_return_conditional_losses_260462
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
d
H__inference_leaky_re_lu_5_layer_call_and_return_conditional_losses_26818

inputs
identityn
	LeakyRelu	LeakyReluinputs*A
_output_shapes/
-:+???????????????????????????2
	LeakyRelu?
IdentityIdentityLeakyRelu:activations:0*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+???????????????????????????:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
d
H__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_26773

inputs
identity\
	LeakyRelu	LeakyReluinputs*/
_output_shapes
:?????????2
	LeakyRelus
IdentityIdentityLeakyRelu:activations:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
C__inference_conv2d_2_layer_call_and_return_conditional_losses_26768

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
C__inference_conv2d_5_layer_call_and_return_conditional_losses_26813

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
f
J__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_26070

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingSAME*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
f
J__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_26082

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingSAME*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
d
H__inference_leaky_re_lu_3_layer_call_and_return_conditional_losses_26234

inputs
identity\
	LeakyRelu	LeakyReluinputs*/
_output_shapes
:?????????2
	LeakyRelus
IdentityIdentityLeakyRelu:activations:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
f
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_26058

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingSAME*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
d
H__inference_leaky_re_lu_4_layer_call_and_return_conditional_losses_26803

inputs
identity\
	LeakyRelu	LeakyReluinputs*/
_output_shapes
:?????????2
	LeakyRelus
IdentityIdentityLeakyRelu:activations:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
C__inference_conv2d_7_layer_call_and_return_conditional_losses_26320

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????2	
BiasAdd{
SigmoidSigmoidBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2	
Sigmoid?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+??????????????????????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
d
H__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_26186

inputs
identity\
	LeakyRelu	LeakyReluinputs*/
_output_shapes
:????????? 2
	LeakyRelus
IdentityIdentityLeakyRelu:activations:0*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
d
H__inference_leaky_re_lu_5_layer_call_and_return_conditional_losses_26282

inputs
identityn
	LeakyRelu	LeakyReluinputs*A
_output_shapes/
-:+???????????????????????????2
	LeakyRelu?
IdentityIdentityLeakyRelu:activations:0*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+???????????????????????????:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
??
?
 __inference__wrapped_model_26039
zero_padding2d_1_inputN
4sequential_1_conv2d_1_conv2d_readvariableop_resource: C
5sequential_1_conv2d_1_biasadd_readvariableop_resource: N
4sequential_1_conv2d_2_conv2d_readvariableop_resource: C
5sequential_1_conv2d_2_biasadd_readvariableop_resource:N
4sequential_1_conv2d_3_conv2d_readvariableop_resource:C
5sequential_1_conv2d_3_biasadd_readvariableop_resource:N
4sequential_1_conv2d_4_conv2d_readvariableop_resource:C
5sequential_1_conv2d_4_biasadd_readvariableop_resource:N
4sequential_1_conv2d_5_conv2d_readvariableop_resource:C
5sequential_1_conv2d_5_biasadd_readvariableop_resource:N
4sequential_1_conv2d_6_conv2d_readvariableop_resource: C
5sequential_1_conv2d_6_biasadd_readvariableop_resource: N
4sequential_1_conv2d_7_conv2d_readvariableop_resource: C
5sequential_1_conv2d_7_biasadd_readvariableop_resource:
identity??,sequential_1/conv2d_1/BiasAdd/ReadVariableOp?+sequential_1/conv2d_1/Conv2D/ReadVariableOp?,sequential_1/conv2d_2/BiasAdd/ReadVariableOp?+sequential_1/conv2d_2/Conv2D/ReadVariableOp?,sequential_1/conv2d_3/BiasAdd/ReadVariableOp?+sequential_1/conv2d_3/Conv2D/ReadVariableOp?,sequential_1/conv2d_4/BiasAdd/ReadVariableOp?+sequential_1/conv2d_4/Conv2D/ReadVariableOp?,sequential_1/conv2d_5/BiasAdd/ReadVariableOp?+sequential_1/conv2d_5/Conv2D/ReadVariableOp?,sequential_1/conv2d_6/BiasAdd/ReadVariableOp?+sequential_1/conv2d_6/Conv2D/ReadVariableOp?,sequential_1/conv2d_7/BiasAdd/ReadVariableOp?+sequential_1/conv2d_7/Conv2D/ReadVariableOp?
*sequential_1/zero_padding2d_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                             2,
*sequential_1/zero_padding2d_1/Pad/paddings?
!sequential_1/zero_padding2d_1/PadPadzero_padding2d_1_input3sequential_1/zero_padding2d_1/Pad/paddings:output:0*
T0*/
_output_shapes
:?????????2#
!sequential_1/zero_padding2d_1/Pad?
+sequential_1/conv2d_1/Conv2D/ReadVariableOpReadVariableOp4sequential_1_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02-
+sequential_1/conv2d_1/Conv2D/ReadVariableOp?
sequential_1/conv2d_1/Conv2DConv2D*sequential_1/zero_padding2d_1/Pad:output:03sequential_1/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2
sequential_1/conv2d_1/Conv2D?
,sequential_1/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp5sequential_1_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,sequential_1/conv2d_1/BiasAdd/ReadVariableOp?
sequential_1/conv2d_1/BiasAddBiasAdd%sequential_1/conv2d_1/Conv2D:output:04sequential_1/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
sequential_1/conv2d_1/BiasAdd?
$sequential_1/leaky_re_lu_1/LeakyRelu	LeakyRelu&sequential_1/conv2d_1/BiasAdd:output:0*/
_output_shapes
:????????? 2&
$sequential_1/leaky_re_lu_1/LeakyRelu?
$sequential_1/max_pooling2d_1/MaxPoolMaxPool2sequential_1/leaky_re_lu_1/LeakyRelu:activations:0*/
_output_shapes
:????????? *
ksize
*
paddingSAME*
strides
2&
$sequential_1/max_pooling2d_1/MaxPool?
+sequential_1/conv2d_2/Conv2D/ReadVariableOpReadVariableOp4sequential_1_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02-
+sequential_1/conv2d_2/Conv2D/ReadVariableOp?
sequential_1/conv2d_2/Conv2DConv2D-sequential_1/max_pooling2d_1/MaxPool:output:03sequential_1/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
sequential_1/conv2d_2/Conv2D?
,sequential_1/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp5sequential_1_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,sequential_1/conv2d_2/BiasAdd/ReadVariableOp?
sequential_1/conv2d_2/BiasAddBiasAdd%sequential_1/conv2d_2/Conv2D:output:04sequential_1/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
sequential_1/conv2d_2/BiasAdd?
$sequential_1/leaky_re_lu_2/LeakyRelu	LeakyRelu&sequential_1/conv2d_2/BiasAdd:output:0*/
_output_shapes
:?????????2&
$sequential_1/leaky_re_lu_2/LeakyRelu?
$sequential_1/max_pooling2d_2/MaxPoolMaxPool2sequential_1/leaky_re_lu_2/LeakyRelu:activations:0*/
_output_shapes
:?????????*
ksize
*
paddingSAME*
strides
2&
$sequential_1/max_pooling2d_2/MaxPool?
+sequential_1/conv2d_3/Conv2D/ReadVariableOpReadVariableOp4sequential_1_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02-
+sequential_1/conv2d_3/Conv2D/ReadVariableOp?
sequential_1/conv2d_3/Conv2DConv2D-sequential_1/max_pooling2d_2/MaxPool:output:03sequential_1/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
sequential_1/conv2d_3/Conv2D?
,sequential_1/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp5sequential_1_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,sequential_1/conv2d_3/BiasAdd/ReadVariableOp?
sequential_1/conv2d_3/BiasAddBiasAdd%sequential_1/conv2d_3/Conv2D:output:04sequential_1/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
sequential_1/conv2d_3/BiasAdd?
$sequential_1/leaky_re_lu_3/LeakyRelu	LeakyRelu&sequential_1/conv2d_3/BiasAdd:output:0*/
_output_shapes
:?????????2&
$sequential_1/leaky_re_lu_3/LeakyRelu?
$sequential_1/max_pooling2d_3/MaxPoolMaxPool2sequential_1/leaky_re_lu_3/LeakyRelu:activations:0*/
_output_shapes
:?????????*
ksize
*
paddingSAME*
strides
2&
$sequential_1/max_pooling2d_3/MaxPool?
+sequential_1/conv2d_4/Conv2D/ReadVariableOpReadVariableOp4sequential_1_conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02-
+sequential_1/conv2d_4/Conv2D/ReadVariableOp?
sequential_1/conv2d_4/Conv2DConv2D-sequential_1/max_pooling2d_3/MaxPool:output:03sequential_1/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
sequential_1/conv2d_4/Conv2D?
,sequential_1/conv2d_4/BiasAdd/ReadVariableOpReadVariableOp5sequential_1_conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,sequential_1/conv2d_4/BiasAdd/ReadVariableOp?
sequential_1/conv2d_4/BiasAddBiasAdd%sequential_1/conv2d_4/Conv2D:output:04sequential_1/conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
sequential_1/conv2d_4/BiasAdd?
$sequential_1/leaky_re_lu_4/LeakyRelu	LeakyRelu&sequential_1/conv2d_4/BiasAdd:output:0*/
_output_shapes
:?????????2&
$sequential_1/leaky_re_lu_4/LeakyRelu?
"sequential_1/up_sampling2d_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2$
"sequential_1/up_sampling2d_1/Const?
$sequential_1/up_sampling2d_1/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      2&
$sequential_1/up_sampling2d_1/Const_1?
 sequential_1/up_sampling2d_1/mulMul+sequential_1/up_sampling2d_1/Const:output:0-sequential_1/up_sampling2d_1/Const_1:output:0*
T0*
_output_shapes
:2"
 sequential_1/up_sampling2d_1/mul?
9sequential_1/up_sampling2d_1/resize/ResizeNearestNeighborResizeNearestNeighbor2sequential_1/leaky_re_lu_4/LeakyRelu:activations:0$sequential_1/up_sampling2d_1/mul:z:0*
T0*/
_output_shapes
:?????????*
half_pixel_centers(2;
9sequential_1/up_sampling2d_1/resize/ResizeNearestNeighbor?
+sequential_1/conv2d_5/Conv2D/ReadVariableOpReadVariableOp4sequential_1_conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02-
+sequential_1/conv2d_5/Conv2D/ReadVariableOp?
sequential_1/conv2d_5/Conv2DConv2DJsequential_1/up_sampling2d_1/resize/ResizeNearestNeighbor:resized_images:03sequential_1/conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
sequential_1/conv2d_5/Conv2D?
,sequential_1/conv2d_5/BiasAdd/ReadVariableOpReadVariableOp5sequential_1_conv2d_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,sequential_1/conv2d_5/BiasAdd/ReadVariableOp?
sequential_1/conv2d_5/BiasAddBiasAdd%sequential_1/conv2d_5/Conv2D:output:04sequential_1/conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
sequential_1/conv2d_5/BiasAdd?
$sequential_1/leaky_re_lu_5/LeakyRelu	LeakyRelu&sequential_1/conv2d_5/BiasAdd:output:0*/
_output_shapes
:?????????2&
$sequential_1/leaky_re_lu_5/LeakyRelu?
"sequential_1/up_sampling2d_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2$
"sequential_1/up_sampling2d_2/Const?
$sequential_1/up_sampling2d_2/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      2&
$sequential_1/up_sampling2d_2/Const_1?
 sequential_1/up_sampling2d_2/mulMul+sequential_1/up_sampling2d_2/Const:output:0-sequential_1/up_sampling2d_2/Const_1:output:0*
T0*
_output_shapes
:2"
 sequential_1/up_sampling2d_2/mul?
9sequential_1/up_sampling2d_2/resize/ResizeNearestNeighborResizeNearestNeighbor2sequential_1/leaky_re_lu_5/LeakyRelu:activations:0$sequential_1/up_sampling2d_2/mul:z:0*
T0*/
_output_shapes
:?????????*
half_pixel_centers(2;
9sequential_1/up_sampling2d_2/resize/ResizeNearestNeighbor?
+sequential_1/conv2d_6/Conv2D/ReadVariableOpReadVariableOp4sequential_1_conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02-
+sequential_1/conv2d_6/Conv2D/ReadVariableOp?
sequential_1/conv2d_6/Conv2DConv2DJsequential_1/up_sampling2d_2/resize/ResizeNearestNeighbor:resized_images:03sequential_1/conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2
sequential_1/conv2d_6/Conv2D?
,sequential_1/conv2d_6/BiasAdd/ReadVariableOpReadVariableOp5sequential_1_conv2d_6_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,sequential_1/conv2d_6/BiasAdd/ReadVariableOp?
sequential_1/conv2d_6/BiasAddBiasAdd%sequential_1/conv2d_6/Conv2D:output:04sequential_1/conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
sequential_1/conv2d_6/BiasAdd?
$sequential_1/leaky_re_lu_6/LeakyRelu	LeakyRelu&sequential_1/conv2d_6/BiasAdd:output:0*/
_output_shapes
:????????? 2&
$sequential_1/leaky_re_lu_6/LeakyRelu?
"sequential_1/up_sampling2d_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2$
"sequential_1/up_sampling2d_3/Const?
$sequential_1/up_sampling2d_3/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      2&
$sequential_1/up_sampling2d_3/Const_1?
 sequential_1/up_sampling2d_3/mulMul+sequential_1/up_sampling2d_3/Const:output:0-sequential_1/up_sampling2d_3/Const_1:output:0*
T0*
_output_shapes
:2"
 sequential_1/up_sampling2d_3/mul?
9sequential_1/up_sampling2d_3/resize/ResizeNearestNeighborResizeNearestNeighbor2sequential_1/leaky_re_lu_6/LeakyRelu:activations:0$sequential_1/up_sampling2d_3/mul:z:0*
T0*/
_output_shapes
:????????? *
half_pixel_centers(2;
9sequential_1/up_sampling2d_3/resize/ResizeNearestNeighbor?
+sequential_1/conv2d_7/Conv2D/ReadVariableOpReadVariableOp4sequential_1_conv2d_7_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02-
+sequential_1/conv2d_7/Conv2D/ReadVariableOp?
sequential_1/conv2d_7/Conv2DConv2DJsequential_1/up_sampling2d_3/resize/ResizeNearestNeighbor:resized_images:03sequential_1/conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
sequential_1/conv2d_7/Conv2D?
,sequential_1/conv2d_7/BiasAdd/ReadVariableOpReadVariableOp5sequential_1_conv2d_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,sequential_1/conv2d_7/BiasAdd/ReadVariableOp?
sequential_1/conv2d_7/BiasAddBiasAdd%sequential_1/conv2d_7/Conv2D:output:04sequential_1/conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
sequential_1/conv2d_7/BiasAdd?
sequential_1/conv2d_7/SigmoidSigmoid&sequential_1/conv2d_7/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
sequential_1/conv2d_7/Sigmoid?
-sequential_1/cropping2d_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"              2/
-sequential_1/cropping2d_1/strided_slice/stack?
/sequential_1/cropping2d_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"    ????????    21
/sequential_1/cropping2d_1/strided_slice/stack_1?
/sequential_1/cropping2d_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            21
/sequential_1/cropping2d_1/strided_slice/stack_2?
'sequential_1/cropping2d_1/strided_sliceStridedSlice!sequential_1/conv2d_7/Sigmoid:y:06sequential_1/cropping2d_1/strided_slice/stack:output:08sequential_1/cropping2d_1/strided_slice/stack_1:output:08sequential_1/cropping2d_1/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:?????????*

begin_mask	*
end_mask	2)
'sequential_1/cropping2d_1/strided_slice?
IdentityIdentity0sequential_1/cropping2d_1/strided_slice:output:0-^sequential_1/conv2d_1/BiasAdd/ReadVariableOp,^sequential_1/conv2d_1/Conv2D/ReadVariableOp-^sequential_1/conv2d_2/BiasAdd/ReadVariableOp,^sequential_1/conv2d_2/Conv2D/ReadVariableOp-^sequential_1/conv2d_3/BiasAdd/ReadVariableOp,^sequential_1/conv2d_3/Conv2D/ReadVariableOp-^sequential_1/conv2d_4/BiasAdd/ReadVariableOp,^sequential_1/conv2d_4/Conv2D/ReadVariableOp-^sequential_1/conv2d_5/BiasAdd/ReadVariableOp,^sequential_1/conv2d_5/Conv2D/ReadVariableOp-^sequential_1/conv2d_6/BiasAdd/ReadVariableOp,^sequential_1/conv2d_6/Conv2D/ReadVariableOp-^sequential_1/conv2d_7/BiasAdd/ReadVariableOp,^sequential_1/conv2d_7/Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:?????????: : : : : : : : : : : : : : 2\
,sequential_1/conv2d_1/BiasAdd/ReadVariableOp,sequential_1/conv2d_1/BiasAdd/ReadVariableOp2Z
+sequential_1/conv2d_1/Conv2D/ReadVariableOp+sequential_1/conv2d_1/Conv2D/ReadVariableOp2\
,sequential_1/conv2d_2/BiasAdd/ReadVariableOp,sequential_1/conv2d_2/BiasAdd/ReadVariableOp2Z
+sequential_1/conv2d_2/Conv2D/ReadVariableOp+sequential_1/conv2d_2/Conv2D/ReadVariableOp2\
,sequential_1/conv2d_3/BiasAdd/ReadVariableOp,sequential_1/conv2d_3/BiasAdd/ReadVariableOp2Z
+sequential_1/conv2d_3/Conv2D/ReadVariableOp+sequential_1/conv2d_3/Conv2D/ReadVariableOp2\
,sequential_1/conv2d_4/BiasAdd/ReadVariableOp,sequential_1/conv2d_4/BiasAdd/ReadVariableOp2Z
+sequential_1/conv2d_4/Conv2D/ReadVariableOp+sequential_1/conv2d_4/Conv2D/ReadVariableOp2\
,sequential_1/conv2d_5/BiasAdd/ReadVariableOp,sequential_1/conv2d_5/BiasAdd/ReadVariableOp2Z
+sequential_1/conv2d_5/Conv2D/ReadVariableOp+sequential_1/conv2d_5/Conv2D/ReadVariableOp2\
,sequential_1/conv2d_6/BiasAdd/ReadVariableOp,sequential_1/conv2d_6/BiasAdd/ReadVariableOp2Z
+sequential_1/conv2d_6/Conv2D/ReadVariableOp+sequential_1/conv2d_6/Conv2D/ReadVariableOp2\
,sequential_1/conv2d_7/BiasAdd/ReadVariableOp,sequential_1/conv2d_7/BiasAdd/ReadVariableOp2Z
+sequential_1/conv2d_7/Conv2D/ReadVariableOp+sequential_1/conv2d_7/Conv2D/ReadVariableOp:g c
/
_output_shapes
:?????????
0
_user_specified_namezero_padding2d_1_input
?

?
C__inference_conv2d_1_layer_call_and_return_conditional_losses_26753

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?f
?
__inference__traced_save_27020
file_prefix.
*savev2_conv2d_1_kernel_read_readvariableop,
(savev2_conv2d_1_bias_read_readvariableop.
*savev2_conv2d_2_kernel_read_readvariableop,
(savev2_conv2d_2_bias_read_readvariableop.
*savev2_conv2d_3_kernel_read_readvariableop,
(savev2_conv2d_3_bias_read_readvariableop.
*savev2_conv2d_4_kernel_read_readvariableop,
(savev2_conv2d_4_bias_read_readvariableop.
*savev2_conv2d_5_kernel_read_readvariableop,
(savev2_conv2d_5_bias_read_readvariableop.
*savev2_conv2d_6_kernel_read_readvariableop,
(savev2_conv2d_6_bias_read_readvariableop.
*savev2_conv2d_7_kernel_read_readvariableop,
(savev2_conv2d_7_bias_read_readvariableop#
savev2_iter_read_readvariableop	%
!savev2_beta_1_read_readvariableop%
!savev2_beta_2_read_readvariableop$
 savev2_decay_read_readvariableop,
(savev2_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop0
,savev2_conv2d_1_kernel_m_read_readvariableop.
*savev2_conv2d_1_bias_m_read_readvariableop0
,savev2_conv2d_2_kernel_m_read_readvariableop.
*savev2_conv2d_2_bias_m_read_readvariableop0
,savev2_conv2d_3_kernel_m_read_readvariableop.
*savev2_conv2d_3_bias_m_read_readvariableop0
,savev2_conv2d_4_kernel_m_read_readvariableop.
*savev2_conv2d_4_bias_m_read_readvariableop0
,savev2_conv2d_5_kernel_m_read_readvariableop.
*savev2_conv2d_5_bias_m_read_readvariableop0
,savev2_conv2d_6_kernel_m_read_readvariableop.
*savev2_conv2d_6_bias_m_read_readvariableop0
,savev2_conv2d_7_kernel_m_read_readvariableop.
*savev2_conv2d_7_bias_m_read_readvariableop0
,savev2_conv2d_1_kernel_v_read_readvariableop.
*savev2_conv2d_1_bias_v_read_readvariableop0
,savev2_conv2d_2_kernel_v_read_readvariableop.
*savev2_conv2d_2_bias_v_read_readvariableop0
,savev2_conv2d_3_kernel_v_read_readvariableop.
*savev2_conv2d_3_bias_v_read_readvariableop0
,savev2_conv2d_4_kernel_v_read_readvariableop.
*savev2_conv2d_4_bias_v_read_readvariableop0
,savev2_conv2d_5_kernel_v_read_readvariableop.
*savev2_conv2d_5_bias_v_read_readvariableop0
,savev2_conv2d_6_kernel_v_read_readvariableop.
*savev2_conv2d_6_bias_v_read_readvariableop0
,savev2_conv2d_7_kernel_v_read_readvariableop.
*savev2_conv2d_7_bias_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:4*
dtype0*?
value?B?4B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:4*
dtype0*{
valuerBp4B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_conv2d_1_kernel_read_readvariableop(savev2_conv2d_1_bias_read_readvariableop*savev2_conv2d_2_kernel_read_readvariableop(savev2_conv2d_2_bias_read_readvariableop*savev2_conv2d_3_kernel_read_readvariableop(savev2_conv2d_3_bias_read_readvariableop*savev2_conv2d_4_kernel_read_readvariableop(savev2_conv2d_4_bias_read_readvariableop*savev2_conv2d_5_kernel_read_readvariableop(savev2_conv2d_5_bias_read_readvariableop*savev2_conv2d_6_kernel_read_readvariableop(savev2_conv2d_6_bias_read_readvariableop*savev2_conv2d_7_kernel_read_readvariableop(savev2_conv2d_7_bias_read_readvariableopsavev2_iter_read_readvariableop!savev2_beta_1_read_readvariableop!savev2_beta_2_read_readvariableop savev2_decay_read_readvariableop(savev2_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop,savev2_conv2d_1_kernel_m_read_readvariableop*savev2_conv2d_1_bias_m_read_readvariableop,savev2_conv2d_2_kernel_m_read_readvariableop*savev2_conv2d_2_bias_m_read_readvariableop,savev2_conv2d_3_kernel_m_read_readvariableop*savev2_conv2d_3_bias_m_read_readvariableop,savev2_conv2d_4_kernel_m_read_readvariableop*savev2_conv2d_4_bias_m_read_readvariableop,savev2_conv2d_5_kernel_m_read_readvariableop*savev2_conv2d_5_bias_m_read_readvariableop,savev2_conv2d_6_kernel_m_read_readvariableop*savev2_conv2d_6_bias_m_read_readvariableop,savev2_conv2d_7_kernel_m_read_readvariableop*savev2_conv2d_7_bias_m_read_readvariableop,savev2_conv2d_1_kernel_v_read_readvariableop*savev2_conv2d_1_bias_v_read_readvariableop,savev2_conv2d_2_kernel_v_read_readvariableop*savev2_conv2d_2_bias_v_read_readvariableop,savev2_conv2d_3_kernel_v_read_readvariableop*savev2_conv2d_3_bias_v_read_readvariableop,savev2_conv2d_4_kernel_v_read_readvariableop*savev2_conv2d_4_bias_v_read_readvariableop,savev2_conv2d_5_kernel_v_read_readvariableop*savev2_conv2d_5_bias_v_read_readvariableop,savev2_conv2d_6_kernel_v_read_readvariableop*savev2_conv2d_6_bias_v_read_readvariableop,savev2_conv2d_7_kernel_v_read_readvariableop*savev2_conv2d_7_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *B
dtypes8
624	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*?
_input_shapes?
?: : : : :::::::: : : :: : : : : : : : : : : : :::::::: : : :: : : :::::::: : : :: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: : 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,	(
&
_output_shapes
:: 


_output_shapes
::,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: : 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: : 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::, (
&
_output_shapes
:: !

_output_shapes
::,"(
&
_output_shapes
: : #

_output_shapes
: :,$(
&
_output_shapes
: : %

_output_shapes
::,&(
&
_output_shapes
: : '

_output_shapes
: :,((
&
_output_shapes
: : )

_output_shapes
::,*(
&
_output_shapes
:: +

_output_shapes
::,,(
&
_output_shapes
:: -

_output_shapes
::,.(
&
_output_shapes
:: /

_output_shapes
::,0(
&
_output_shapes
: : 1

_output_shapes
: :,2(
&
_output_shapes
: : 3

_output_shapes
::4

_output_shapes
: 
?
d
H__inference_leaky_re_lu_6_layer_call_and_return_conditional_losses_26833

inputs
identityn
	LeakyRelu	LeakyReluinputs*A
_output_shapes/
-:+??????????????????????????? 2
	LeakyRelu?
IdentityIdentityLeakyRelu:activations:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+??????????????????????????? :i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?

?
C__inference_conv2d_3_layer_call_and_return_conditional_losses_26783

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
f
J__inference_up_sampling2d_3_layer_call_and_return_conditional_losses_26139

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
strided_slice_
ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
Const^
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:2
mul?
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4????????????????????????????????????*
half_pixel_centers(2
resize/ResizeNearestNeighbor?
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
??
?
!__inference__traced_restore_27183
file_prefix:
 assignvariableop_conv2d_1_kernel: .
 assignvariableop_1_conv2d_1_bias: <
"assignvariableop_2_conv2d_2_kernel: .
 assignvariableop_3_conv2d_2_bias:<
"assignvariableop_4_conv2d_3_kernel:.
 assignvariableop_5_conv2d_3_bias:<
"assignvariableop_6_conv2d_4_kernel:.
 assignvariableop_7_conv2d_4_bias:<
"assignvariableop_8_conv2d_5_kernel:.
 assignvariableop_9_conv2d_5_bias:=
#assignvariableop_10_conv2d_6_kernel: /
!assignvariableop_11_conv2d_6_bias: =
#assignvariableop_12_conv2d_7_kernel: /
!assignvariableop_13_conv2d_7_bias:"
assignvariableop_14_iter:	 $
assignvariableop_15_beta_1: $
assignvariableop_16_beta_2: #
assignvariableop_17_decay: +
!assignvariableop_18_learning_rate: #
assignvariableop_19_total: #
assignvariableop_20_count: %
assignvariableop_21_total_1: %
assignvariableop_22_count_1: ?
%assignvariableop_23_conv2d_1_kernel_m: 1
#assignvariableop_24_conv2d_1_bias_m: ?
%assignvariableop_25_conv2d_2_kernel_m: 1
#assignvariableop_26_conv2d_2_bias_m:?
%assignvariableop_27_conv2d_3_kernel_m:1
#assignvariableop_28_conv2d_3_bias_m:?
%assignvariableop_29_conv2d_4_kernel_m:1
#assignvariableop_30_conv2d_4_bias_m:?
%assignvariableop_31_conv2d_5_kernel_m:1
#assignvariableop_32_conv2d_5_bias_m:?
%assignvariableop_33_conv2d_6_kernel_m: 1
#assignvariableop_34_conv2d_6_bias_m: ?
%assignvariableop_35_conv2d_7_kernel_m: 1
#assignvariableop_36_conv2d_7_bias_m:?
%assignvariableop_37_conv2d_1_kernel_v: 1
#assignvariableop_38_conv2d_1_bias_v: ?
%assignvariableop_39_conv2d_2_kernel_v: 1
#assignvariableop_40_conv2d_2_bias_v:?
%assignvariableop_41_conv2d_3_kernel_v:1
#assignvariableop_42_conv2d_3_bias_v:?
%assignvariableop_43_conv2d_4_kernel_v:1
#assignvariableop_44_conv2d_4_bias_v:?
%assignvariableop_45_conv2d_5_kernel_v:1
#assignvariableop_46_conv2d_5_bias_v:?
%assignvariableop_47_conv2d_6_kernel_v: 1
#assignvariableop_48_conv2d_6_bias_v: ?
%assignvariableop_49_conv2d_7_kernel_v: 1
#assignvariableop_50_conv2d_7_bias_v:
identity_52??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_49?AssignVariableOp_5?AssignVariableOp_50?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:4*
dtype0*?
value?B?4B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:4*
dtype0*{
valuerBp4B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::::::::::*B
dtypes8
624	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp assignvariableop_conv2d_1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp assignvariableop_1_conv2d_1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp"assignvariableop_2_conv2d_2_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp assignvariableop_3_conv2d_2_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp"assignvariableop_4_conv2d_3_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp assignvariableop_5_conv2d_3_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp"assignvariableop_6_conv2d_4_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp assignvariableop_7_conv2d_4_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp"assignvariableop_8_conv2d_5_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp assignvariableop_9_conv2d_5_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp#assignvariableop_10_conv2d_6_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp!assignvariableop_11_conv2d_6_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp#assignvariableop_12_conv2d_7_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp!assignvariableop_13_conv2d_7_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOpassignvariableop_14_iterIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOpassignvariableop_15_beta_1Identity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOpassignvariableop_16_beta_2Identity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOpassignvariableop_17_decayIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp!assignvariableop_18_learning_rateIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOpassignvariableop_19_totalIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOpassignvariableop_20_countIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOpassignvariableop_21_total_1Identity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOpassignvariableop_22_count_1Identity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp%assignvariableop_23_conv2d_1_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp#assignvariableop_24_conv2d_1_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp%assignvariableop_25_conv2d_2_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp#assignvariableop_26_conv2d_2_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp%assignvariableop_27_conv2d_3_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp#assignvariableop_28_conv2d_3_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp%assignvariableop_29_conv2d_4_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp#assignvariableop_30_conv2d_4_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp%assignvariableop_31_conv2d_5_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp#assignvariableop_32_conv2d_5_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp%assignvariableop_33_conv2d_6_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp#assignvariableop_34_conv2d_6_bias_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOp%assignvariableop_35_conv2d_7_kernel_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOp#assignvariableop_36_conv2d_7_bias_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOp%assignvariableop_37_conv2d_1_kernel_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOp#assignvariableop_38_conv2d_1_bias_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOp%assignvariableop_39_conv2d_2_kernel_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOp#assignvariableop_40_conv2d_2_bias_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41?
AssignVariableOp_41AssignVariableOp%assignvariableop_41_conv2d_3_kernel_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42?
AssignVariableOp_42AssignVariableOp#assignvariableop_42_conv2d_3_bias_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43?
AssignVariableOp_43AssignVariableOp%assignvariableop_43_conv2d_4_kernel_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44?
AssignVariableOp_44AssignVariableOp#assignvariableop_44_conv2d_4_bias_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45?
AssignVariableOp_45AssignVariableOp%assignvariableop_45_conv2d_5_kernel_vIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46?
AssignVariableOp_46AssignVariableOp#assignvariableop_46_conv2d_5_bias_vIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47?
AssignVariableOp_47AssignVariableOp%assignvariableop_47_conv2d_6_kernel_vIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48?
AssignVariableOp_48AssignVariableOp#assignvariableop_48_conv2d_6_bias_vIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49?
AssignVariableOp_49AssignVariableOp%assignvariableop_49_conv2d_7_kernel_vIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50?
AssignVariableOp_50AssignVariableOp#assignvariableop_50_conv2d_7_bias_vIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_509
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?	
Identity_51Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_51?	
Identity_52IdentityIdentity_51:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_52"#
identity_52Identity_52:output:0*{
_input_shapesj
h: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?	
c
G__inference_cropping2d_1_layer_call_and_return_conditional_losses_26154

inputs
identity?
strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"              2
strided_slice/stack?
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"    ????????    2
strided_slice/stack_1?
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2
strided_slice/stack_2?
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*J
_output_shapes8
6:4????????????????????????????????????*

begin_mask	*
end_mask	2
strided_slice?
IdentityIdentitystrided_slice:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?

?
C__inference_conv2d_3_layer_call_and_return_conditional_losses_26223

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
d
H__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_26758

inputs
identity\
	LeakyRelu	LeakyReluinputs*/
_output_shapes
:????????? 2
	LeakyRelus
IdentityIdentityLeakyRelu:activations:0*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
d
H__inference_leaky_re_lu_3_layer_call_and_return_conditional_losses_26788

inputs
identity\
	LeakyRelu	LeakyReluinputs*/
_output_shapes
:?????????2
	LeakyRelus
IdentityIdentityLeakyRelu:activations:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
d
H__inference_leaky_re_lu_4_layer_call_and_return_conditional_losses_26258

inputs
identity\
	LeakyRelu	LeakyReluinputs*/
_output_shapes
:?????????2
	LeakyRelus
IdentityIdentityLeakyRelu:activations:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?T
?
G__inference_sequential_1_layer_call_and_return_conditional_losses_26437

inputs(
conv2d_1_26388: 
conv2d_1_26390: (
conv2d_2_26395: 
conv2d_2_26397:(
conv2d_3_26402:
conv2d_3_26404:(
conv2d_4_26409:
conv2d_4_26411:(
conv2d_5_26416:
conv2d_5_26418:(
conv2d_6_26423: 
conv2d_6_26425: (
conv2d_7_26430: 
conv2d_7_26432:
identity?? conv2d_1/StatefulPartitionedCall? conv2d_2/StatefulPartitionedCall? conv2d_3/StatefulPartitionedCall? conv2d_4/StatefulPartitionedCall? conv2d_5/StatefulPartitionedCall? conv2d_6/StatefulPartitionedCall? conv2d_7/StatefulPartitionedCall?
 zero_padding2d_1/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_zero_padding2d_1_layer_call_and_return_conditional_losses_260462"
 zero_padding2d_1/PartitionedCall?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall)zero_padding2d_1/PartitionedCall:output:0conv2d_1_26388conv2d_1_26390*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_261752"
 conv2d_1/StatefulPartitionedCall?
leaky_re_lu_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_261862
leaky_re_lu_1/PartitionedCall?
max_pooling2d_1/PartitionedCallPartitionedCall&leaky_re_lu_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_260582!
max_pooling2d_1/PartitionedCall?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0conv2d_2_26395conv2d_2_26397*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_2_layer_call_and_return_conditional_losses_261992"
 conv2d_2/StatefulPartitionedCall?
leaky_re_lu_2/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_262102
leaky_re_lu_2/PartitionedCall?
max_pooling2d_2/PartitionedCallPartitionedCall&leaky_re_lu_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_260702!
max_pooling2d_2/PartitionedCall?
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_2/PartitionedCall:output:0conv2d_3_26402conv2d_3_26404*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_3_layer_call_and_return_conditional_losses_262232"
 conv2d_3/StatefulPartitionedCall?
leaky_re_lu_3/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_leaky_re_lu_3_layer_call_and_return_conditional_losses_262342
leaky_re_lu_3/PartitionedCall?
max_pooling2d_3/PartitionedCallPartitionedCall&leaky_re_lu_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_260822!
max_pooling2d_3/PartitionedCall?
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_3/PartitionedCall:output:0conv2d_4_26409conv2d_4_26411*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_4_layer_call_and_return_conditional_losses_262472"
 conv2d_4/StatefulPartitionedCall?
leaky_re_lu_4/PartitionedCallPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_leaky_re_lu_4_layer_call_and_return_conditional_losses_262582
leaky_re_lu_4/PartitionedCall?
up_sampling2d_1/PartitionedCallPartitionedCall&leaky_re_lu_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_261012!
up_sampling2d_1/PartitionedCall?
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_1/PartitionedCall:output:0conv2d_5_26416conv2d_5_26418*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_5_layer_call_and_return_conditional_losses_262712"
 conv2d_5/StatefulPartitionedCall?
leaky_re_lu_5/PartitionedCallPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_leaky_re_lu_5_layer_call_and_return_conditional_losses_262822
leaky_re_lu_5/PartitionedCall?
up_sampling2d_2/PartitionedCallPartitionedCall&leaky_re_lu_5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_up_sampling2d_2_layer_call_and_return_conditional_losses_261202!
up_sampling2d_2/PartitionedCall?
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_2/PartitionedCall:output:0conv2d_6_26423conv2d_6_26425*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_6_layer_call_and_return_conditional_losses_262952"
 conv2d_6/StatefulPartitionedCall?
leaky_re_lu_6/PartitionedCallPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_leaky_re_lu_6_layer_call_and_return_conditional_losses_263062
leaky_re_lu_6/PartitionedCall?
up_sampling2d_3/PartitionedCallPartitionedCall&leaky_re_lu_6/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_up_sampling2d_3_layer_call_and_return_conditional_losses_261392!
up_sampling2d_3/PartitionedCall?
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_3/PartitionedCall:output:0conv2d_7_26430conv2d_7_26432*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_7_layer_call_and_return_conditional_losses_263202"
 conv2d_7/StatefulPartitionedCall?
cropping2d_1/PartitionedCallPartitionedCall)conv2d_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_cropping2d_1_layer_call_and_return_conditional_losses_261542
cropping2d_1/PartitionedCall?
IdentityIdentity%cropping2d_1/PartitionedCall:output:0!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall!^conv2d_7/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:?????????: : : : : : : : : : : : : : 2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
d
H__inference_leaky_re_lu_6_layer_call_and_return_conditional_losses_26306

inputs
identityn
	LeakyRelu	LeakyReluinputs*A
_output_shapes/
-:+??????????????????????????? 2
	LeakyRelu?
IdentityIdentityLeakyRelu:activations:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+??????????????????????????? :i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
f
J__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_26101

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
strided_slice_
ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
Const^
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:2
mul?
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4????????????????????????????????????*
half_pixel_centers(2
resize/ResizeNearestNeighbor?
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
H
,__inference_cropping2d_1_layer_call_fn_26160

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_cropping2d_1_layer_call_and_return_conditional_losses_261542
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?U
?
G__inference_sequential_1_layer_call_and_return_conditional_losses_26328
zero_padding2d_1_input(
conv2d_1_26176: 
conv2d_1_26178: (
conv2d_2_26200: 
conv2d_2_26202:(
conv2d_3_26224:
conv2d_3_26226:(
conv2d_4_26248:
conv2d_4_26250:(
conv2d_5_26272:
conv2d_5_26274:(
conv2d_6_26296: 
conv2d_6_26298: (
conv2d_7_26321: 
conv2d_7_26323:
identity?? conv2d_1/StatefulPartitionedCall? conv2d_2/StatefulPartitionedCall? conv2d_3/StatefulPartitionedCall? conv2d_4/StatefulPartitionedCall? conv2d_5/StatefulPartitionedCall? conv2d_6/StatefulPartitionedCall? conv2d_7/StatefulPartitionedCall?
 zero_padding2d_1/PartitionedCallPartitionedCallzero_padding2d_1_input*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_zero_padding2d_1_layer_call_and_return_conditional_losses_260462"
 zero_padding2d_1/PartitionedCall?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall)zero_padding2d_1/PartitionedCall:output:0conv2d_1_26176conv2d_1_26178*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_261752"
 conv2d_1/StatefulPartitionedCall?
leaky_re_lu_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_261862
leaky_re_lu_1/PartitionedCall?
max_pooling2d_1/PartitionedCallPartitionedCall&leaky_re_lu_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_260582!
max_pooling2d_1/PartitionedCall?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0conv2d_2_26200conv2d_2_26202*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_2_layer_call_and_return_conditional_losses_261992"
 conv2d_2/StatefulPartitionedCall?
leaky_re_lu_2/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_262102
leaky_re_lu_2/PartitionedCall?
max_pooling2d_2/PartitionedCallPartitionedCall&leaky_re_lu_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_260702!
max_pooling2d_2/PartitionedCall?
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_2/PartitionedCall:output:0conv2d_3_26224conv2d_3_26226*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_3_layer_call_and_return_conditional_losses_262232"
 conv2d_3/StatefulPartitionedCall?
leaky_re_lu_3/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_leaky_re_lu_3_layer_call_and_return_conditional_losses_262342
leaky_re_lu_3/PartitionedCall?
max_pooling2d_3/PartitionedCallPartitionedCall&leaky_re_lu_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_260822!
max_pooling2d_3/PartitionedCall?
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_3/PartitionedCall:output:0conv2d_4_26248conv2d_4_26250*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_4_layer_call_and_return_conditional_losses_262472"
 conv2d_4/StatefulPartitionedCall?
leaky_re_lu_4/PartitionedCallPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_leaky_re_lu_4_layer_call_and_return_conditional_losses_262582
leaky_re_lu_4/PartitionedCall?
up_sampling2d_1/PartitionedCallPartitionedCall&leaky_re_lu_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_261012!
up_sampling2d_1/PartitionedCall?
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_1/PartitionedCall:output:0conv2d_5_26272conv2d_5_26274*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_5_layer_call_and_return_conditional_losses_262712"
 conv2d_5/StatefulPartitionedCall?
leaky_re_lu_5/PartitionedCallPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_leaky_re_lu_5_layer_call_and_return_conditional_losses_262822
leaky_re_lu_5/PartitionedCall?
up_sampling2d_2/PartitionedCallPartitionedCall&leaky_re_lu_5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_up_sampling2d_2_layer_call_and_return_conditional_losses_261202!
up_sampling2d_2/PartitionedCall?
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_2/PartitionedCall:output:0conv2d_6_26296conv2d_6_26298*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_6_layer_call_and_return_conditional_losses_262952"
 conv2d_6/StatefulPartitionedCall?
leaky_re_lu_6/PartitionedCallPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_leaky_re_lu_6_layer_call_and_return_conditional_losses_263062
leaky_re_lu_6/PartitionedCall?
up_sampling2d_3/PartitionedCallPartitionedCall&leaky_re_lu_6/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_up_sampling2d_3_layer_call_and_return_conditional_losses_261392!
up_sampling2d_3/PartitionedCall?
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_3/PartitionedCall:output:0conv2d_7_26321conv2d_7_26323*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_7_layer_call_and_return_conditional_losses_263202"
 conv2d_7/StatefulPartitionedCall?
cropping2d_1/PartitionedCallPartitionedCall)conv2d_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_cropping2d_1_layer_call_and_return_conditional_losses_261542
cropping2d_1/PartitionedCall?
IdentityIdentity%cropping2d_1/PartitionedCall:output:0!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall!^conv2d_7/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:?????????: : : : : : : : : : : : : : 2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall:g c
/
_output_shapes
:?????????
0
_user_specified_namezero_padding2d_1_input
?
?
C__inference_conv2d_6_layer_call_and_return_conditional_losses_26828

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+??????????????????????????? *
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
g
K__inference_zero_padding2d_1_layer_call_and_return_conditional_losses_26046

inputs
identity?
Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                             2
Pad/paddings?
PadPadinputsPad/paddings:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2
Pad?
IdentityIdentityPad:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
d
H__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_26210

inputs
identity\
	LeakyRelu	LeakyReluinputs*/
_output_shapes
:?????????2
	LeakyRelus
IdentityIdentityLeakyRelu:activations:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
K
/__inference_max_pooling2d_3_layer_call_fn_26088

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_260822
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?h
?
G__inference_sequential_1_layer_call_and_return_conditional_losses_26743

inputsA
'conv2d_1_conv2d_readvariableop_resource: 6
(conv2d_1_biasadd_readvariableop_resource: A
'conv2d_2_conv2d_readvariableop_resource: 6
(conv2d_2_biasadd_readvariableop_resource:A
'conv2d_3_conv2d_readvariableop_resource:6
(conv2d_3_biasadd_readvariableop_resource:A
'conv2d_4_conv2d_readvariableop_resource:6
(conv2d_4_biasadd_readvariableop_resource:A
'conv2d_5_conv2d_readvariableop_resource:6
(conv2d_5_biasadd_readvariableop_resource:A
'conv2d_6_conv2d_readvariableop_resource: 6
(conv2d_6_biasadd_readvariableop_resource: A
'conv2d_7_conv2d_readvariableop_resource: 6
(conv2d_7_biasadd_readvariableop_resource:
identity??conv2d_1/BiasAdd/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOp?conv2d_2/BiasAdd/ReadVariableOp?conv2d_2/Conv2D/ReadVariableOp?conv2d_3/BiasAdd/ReadVariableOp?conv2d_3/Conv2D/ReadVariableOp?conv2d_4/BiasAdd/ReadVariableOp?conv2d_4/Conv2D/ReadVariableOp?conv2d_5/BiasAdd/ReadVariableOp?conv2d_5/Conv2D/ReadVariableOp?conv2d_6/BiasAdd/ReadVariableOp?conv2d_6/Conv2D/ReadVariableOp?conv2d_7/BiasAdd/ReadVariableOp?conv2d_7/Conv2D/ReadVariableOp?
zero_padding2d_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                             2
zero_padding2d_1/Pad/paddings?
zero_padding2d_1/PadPadinputs&zero_padding2d_1/Pad/paddings:output:0*
T0*/
_output_shapes
:?????????2
zero_padding2d_1/Pad?
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02 
conv2d_1/Conv2D/ReadVariableOp?
conv2d_1/Conv2DConv2Dzero_padding2d_1/Pad:output:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2
conv2d_1/Conv2D?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_1/BiasAdd/ReadVariableOp?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
conv2d_1/BiasAdd?
leaky_re_lu_1/LeakyRelu	LeakyReluconv2d_1/BiasAdd:output:0*/
_output_shapes
:????????? 2
leaky_re_lu_1/LeakyRelu?
max_pooling2d_1/MaxPoolMaxPool%leaky_re_lu_1/LeakyRelu:activations:0*/
_output_shapes
:????????? *
ksize
*
paddingSAME*
strides
2
max_pooling2d_1/MaxPool?
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02 
conv2d_2/Conv2D/ReadVariableOp?
conv2d_2/Conv2DConv2D max_pooling2d_1/MaxPool:output:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
conv2d_2/Conv2D?
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_2/BiasAdd/ReadVariableOp?
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
conv2d_2/BiasAdd?
leaky_re_lu_2/LeakyRelu	LeakyReluconv2d_2/BiasAdd:output:0*/
_output_shapes
:?????????2
leaky_re_lu_2/LeakyRelu?
max_pooling2d_2/MaxPoolMaxPool%leaky_re_lu_2/LeakyRelu:activations:0*/
_output_shapes
:?????????*
ksize
*
paddingSAME*
strides
2
max_pooling2d_2/MaxPool?
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02 
conv2d_3/Conv2D/ReadVariableOp?
conv2d_3/Conv2DConv2D max_pooling2d_2/MaxPool:output:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
conv2d_3/Conv2D?
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_3/BiasAdd/ReadVariableOp?
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
conv2d_3/BiasAdd?
leaky_re_lu_3/LeakyRelu	LeakyReluconv2d_3/BiasAdd:output:0*/
_output_shapes
:?????????2
leaky_re_lu_3/LeakyRelu?
max_pooling2d_3/MaxPoolMaxPool%leaky_re_lu_3/LeakyRelu:activations:0*/
_output_shapes
:?????????*
ksize
*
paddingSAME*
strides
2
max_pooling2d_3/MaxPool?
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02 
conv2d_4/Conv2D/ReadVariableOp?
conv2d_4/Conv2DConv2D max_pooling2d_3/MaxPool:output:0&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
conv2d_4/Conv2D?
conv2d_4/BiasAdd/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_4/BiasAdd/ReadVariableOp?
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0'conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
conv2d_4/BiasAdd?
leaky_re_lu_4/LeakyRelu	LeakyReluconv2d_4/BiasAdd:output:0*/
_output_shapes
:?????????2
leaky_re_lu_4/LeakyRelu
up_sampling2d_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_1/Const?
up_sampling2d_1/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_1/Const_1?
up_sampling2d_1/mulMulup_sampling2d_1/Const:output:0 up_sampling2d_1/Const_1:output:0*
T0*
_output_shapes
:2
up_sampling2d_1/mul?
,up_sampling2d_1/resize/ResizeNearestNeighborResizeNearestNeighbor%leaky_re_lu_4/LeakyRelu:activations:0up_sampling2d_1/mul:z:0*
T0*/
_output_shapes
:?????????*
half_pixel_centers(2.
,up_sampling2d_1/resize/ResizeNearestNeighbor?
conv2d_5/Conv2D/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02 
conv2d_5/Conv2D/ReadVariableOp?
conv2d_5/Conv2DConv2D=up_sampling2d_1/resize/ResizeNearestNeighbor:resized_images:0&conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
conv2d_5/Conv2D?
conv2d_5/BiasAdd/ReadVariableOpReadVariableOp(conv2d_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_5/BiasAdd/ReadVariableOp?
conv2d_5/BiasAddBiasAddconv2d_5/Conv2D:output:0'conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
conv2d_5/BiasAdd?
leaky_re_lu_5/LeakyRelu	LeakyReluconv2d_5/BiasAdd:output:0*/
_output_shapes
:?????????2
leaky_re_lu_5/LeakyRelu
up_sampling2d_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_2/Const?
up_sampling2d_2/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_2/Const_1?
up_sampling2d_2/mulMulup_sampling2d_2/Const:output:0 up_sampling2d_2/Const_1:output:0*
T0*
_output_shapes
:2
up_sampling2d_2/mul?
,up_sampling2d_2/resize/ResizeNearestNeighborResizeNearestNeighbor%leaky_re_lu_5/LeakyRelu:activations:0up_sampling2d_2/mul:z:0*
T0*/
_output_shapes
:?????????*
half_pixel_centers(2.
,up_sampling2d_2/resize/ResizeNearestNeighbor?
conv2d_6/Conv2D/ReadVariableOpReadVariableOp'conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02 
conv2d_6/Conv2D/ReadVariableOp?
conv2d_6/Conv2DConv2D=up_sampling2d_2/resize/ResizeNearestNeighbor:resized_images:0&conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2
conv2d_6/Conv2D?
conv2d_6/BiasAdd/ReadVariableOpReadVariableOp(conv2d_6_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_6/BiasAdd/ReadVariableOp?
conv2d_6/BiasAddBiasAddconv2d_6/Conv2D:output:0'conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
conv2d_6/BiasAdd?
leaky_re_lu_6/LeakyRelu	LeakyReluconv2d_6/BiasAdd:output:0*/
_output_shapes
:????????? 2
leaky_re_lu_6/LeakyRelu
up_sampling2d_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_3/Const?
up_sampling2d_3/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_3/Const_1?
up_sampling2d_3/mulMulup_sampling2d_3/Const:output:0 up_sampling2d_3/Const_1:output:0*
T0*
_output_shapes
:2
up_sampling2d_3/mul?
,up_sampling2d_3/resize/ResizeNearestNeighborResizeNearestNeighbor%leaky_re_lu_6/LeakyRelu:activations:0up_sampling2d_3/mul:z:0*
T0*/
_output_shapes
:????????? *
half_pixel_centers(2.
,up_sampling2d_3/resize/ResizeNearestNeighbor?
conv2d_7/Conv2D/ReadVariableOpReadVariableOp'conv2d_7_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02 
conv2d_7/Conv2D/ReadVariableOp?
conv2d_7/Conv2DConv2D=up_sampling2d_3/resize/ResizeNearestNeighbor:resized_images:0&conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
conv2d_7/Conv2D?
conv2d_7/BiasAdd/ReadVariableOpReadVariableOp(conv2d_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_7/BiasAdd/ReadVariableOp?
conv2d_7/BiasAddBiasAddconv2d_7/Conv2D:output:0'conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
conv2d_7/BiasAdd?
conv2d_7/SigmoidSigmoidconv2d_7/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
conv2d_7/Sigmoid?
 cropping2d_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"              2"
 cropping2d_1/strided_slice/stack?
"cropping2d_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"    ????????    2$
"cropping2d_1/strided_slice/stack_1?
"cropping2d_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2$
"cropping2d_1/strided_slice/stack_2?
cropping2d_1/strided_sliceStridedSliceconv2d_7/Sigmoid:y:0)cropping2d_1/strided_slice/stack:output:0+cropping2d_1/strided_slice/stack_1:output:0+cropping2d_1/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:?????????*

begin_mask	*
end_mask	2
cropping2d_1/strided_slice?
IdentityIdentity#cropping2d_1/strided_slice:output:0 ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp ^conv2d_4/BiasAdd/ReadVariableOp^conv2d_4/Conv2D/ReadVariableOp ^conv2d_5/BiasAdd/ReadVariableOp^conv2d_5/Conv2D/ReadVariableOp ^conv2d_6/BiasAdd/ReadVariableOp^conv2d_6/Conv2D/ReadVariableOp ^conv2d_7/BiasAdd/ReadVariableOp^conv2d_7/Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:?????????: : : : : : : : : : : : : : 2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp2B
conv2d_4/BiasAdd/ReadVariableOpconv2d_4/BiasAdd/ReadVariableOp2@
conv2d_4/Conv2D/ReadVariableOpconv2d_4/Conv2D/ReadVariableOp2B
conv2d_5/BiasAdd/ReadVariableOpconv2d_5/BiasAdd/ReadVariableOp2@
conv2d_5/Conv2D/ReadVariableOpconv2d_5/Conv2D/ReadVariableOp2B
conv2d_6/BiasAdd/ReadVariableOpconv2d_6/BiasAdd/ReadVariableOp2@
conv2d_6/Conv2D/ReadVariableOpconv2d_6/Conv2D/ReadVariableOp2B
conv2d_7/BiasAdd/ReadVariableOpconv2d_7/BiasAdd/ReadVariableOp2@
conv2d_7/Conv2D/ReadVariableOpconv2d_7/Conv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
K
/__inference_up_sampling2d_1_layer_call_fn_26107

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_261012
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
K
/__inference_up_sampling2d_3_layer_call_fn_26145

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_up_sampling2d_3_layer_call_and_return_conditional_losses_261392
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
K
/__inference_up_sampling2d_2_layer_call_fn_26126

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_up_sampling2d_2_layer_call_and_return_conditional_losses_261202
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
a
zero_padding2d_1_inputG
(serving_default_zero_padding2d_1_input:0?????????H
cropping2d_18
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
??
layer-0
layer_with_weights-0
layer-1
layer-2
layer-3
layer_with_weights-1
layer-4
layer-5
layer-6
layer_with_weights-2
layer-7
	layer-8

layer-9
layer_with_weights-3
layer-10
layer-11
layer-12
layer_with_weights-4
layer-13
layer-14
layer-15
layer_with_weights-5
layer-16
layer-17
layer-18
layer_with_weights-6
layer-19
layer-20
	optimizer
regularization_losses
	variables
trainable_variables
	keras_api

signatures
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses"??
_tf_keras_sequential??{"name": "sequential_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Sequential", "config": {"name": "sequential_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 20, 20, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "zero_padding2d_1_input"}}, {"class_name": "ZeroPadding2D", "config": {"name": "zero_padding2d_1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 20, 20, 1]}, "dtype": "float32", "padding": {"class_name": "__tuple__", "items": [{"class_name": "__tuple__", "items": [2, 2]}, {"class_name": "__tuple__", "items": [2, 2]}]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_avg", "distribution": "uniform", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_1", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_avg", "distribution": "uniform", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_2", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_3", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_avg", "distribution": "uniform", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_3", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_3", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_4", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_avg", "distribution": "uniform", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_4", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}, {"class_name": "UpSampling2D", "config": {"name": "up_sampling2d_1", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_5", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_avg", "distribution": "uniform", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_5", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}, {"class_name": "UpSampling2D", "config": {"name": "up_sampling2d_2", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_6", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_avg", "distribution": "uniform", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_6", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}, {"class_name": "UpSampling2D", "config": {"name": "up_sampling2d_3", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_7", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_avg", "distribution": "uniform", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Cropping2D", "config": {"name": "cropping2d_1", "trainable": true, "dtype": "float32", "cropping": {"class_name": "__tuple__", "items": [{"class_name": "__tuple__", "items": [2, 2]}, {"class_name": "__tuple__", "items": [2, 2]}]}, "data_format": "channels_last"}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 20, 20, 1]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 20, 20, 1]}, "float32", "zero_padding2d_1_input"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 20, 20, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "zero_padding2d_1_input"}}, {"class_name": "ZeroPadding2D", "config": {"name": "zero_padding2d_1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 20, 20, 1]}, "dtype": "float32", "padding": {"class_name": "__tuple__", "items": [{"class_name": "__tuple__", "items": [2, 2]}, {"class_name": "__tuple__", "items": [2, 2]}]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_avg", "distribution": "uniform", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_1", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_avg", "distribution": "uniform", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_2", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_3", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_avg", "distribution": "uniform", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_3", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_3", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_4", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_avg", "distribution": "uniform", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_4", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}, {"class_name": "UpSampling2D", "config": {"name": "up_sampling2d_1", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_5", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_avg", "distribution": "uniform", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_5", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}, {"class_name": "UpSampling2D", "config": {"name": "up_sampling2d_2", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_6", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_avg", "distribution": "uniform", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_6", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}, {"class_name": "UpSampling2D", "config": {"name": "up_sampling2d_3", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_7", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_avg", "distribution": "uniform", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Cropping2D", "config": {"name": "cropping2d_1", "trainable": true, "dtype": "float32", "cropping": {"class_name": "__tuple__", "items": [{"class_name": "__tuple__", "items": [2, 2]}, {"class_name": "__tuple__", "items": [2, 2]}]}, "data_format": "channels_last"}}]}}, "training_config": {"loss": "mean_squared_error", "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "accuracy", "dtype": "float32", "fn": "binary_accuracy"}}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.00019999999494757503, "decay": 0.0, "beta_1": 0.5, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?
regularization_losses
	variables
trainable_variables
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "zero_padding2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 20, 20, 1]}, "stateful": false, "must_restore_from_config": false, "class_name": "ZeroPadding2D", "config": {"name": "zero_padding2d_1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 20, 20, 1]}, "dtype": "float32", "padding": {"class_name": "__tuple__", "items": [{"class_name": "__tuple__", "items": [2, 2]}, {"class_name": "__tuple__", "items": [2, 2]}]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?


 kernel
!bias
"regularization_losses
#	variables
$trainable_variables
%	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?{"name": "conv2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_avg", "distribution": "uniform", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 24, 24, 1]}}
?
&regularization_losses
'	variables
(trainable_variables
)	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "leaky_re_lu_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_1", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}
?
*regularization_losses
+	variables
,trainable_variables
-	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "max_pooling2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?


.kernel
/bias
0regularization_losses
1	variables
2trainable_variables
3	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?{"name": "conv2d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_avg", "distribution": "uniform", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 12, 12, 32]}}
?
4regularization_losses
5	variables
6trainable_variables
7	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "leaky_re_lu_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_2", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}
?
8regularization_losses
9	variables
:trainable_variables
;	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "max_pooling2d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?


<kernel
=bias
>regularization_losses
?	variables
@trainable_variables
A	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?{"name": "conv2d_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_3", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_avg", "distribution": "uniform", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 6, 6, 16]}}
?
Bregularization_losses
C	variables
Dtrainable_variables
E	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "leaky_re_lu_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_3", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}
?
Fregularization_losses
G	variables
Htrainable_variables
I	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "max_pooling2d_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_3", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?


Jkernel
Kbias
Lregularization_losses
M	variables
Ntrainable_variables
O	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?{"name": "conv2d_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_4", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_avg", "distribution": "uniform", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 3, 3, 16]}}
?
Pregularization_losses
Q	variables
Rtrainable_variables
S	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "leaky_re_lu_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_4", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}
?
Tregularization_losses
U	variables
Vtrainable_variables
W	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "up_sampling2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "UpSampling2D", "config": {"name": "up_sampling2d_1", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?


Xkernel
Ybias
Zregularization_losses
[	variables
\trainable_variables
]	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?{"name": "conv2d_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_5", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_avg", "distribution": "uniform", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 6, 6, 16]}}
?
^regularization_losses
_	variables
`trainable_variables
a	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "leaky_re_lu_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_5", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}
?
bregularization_losses
c	variables
dtrainable_variables
e	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "up_sampling2d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "UpSampling2D", "config": {"name": "up_sampling2d_2", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?


fkernel
gbias
hregularization_losses
i	variables
jtrainable_variables
k	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?{"name": "conv2d_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_6", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_avg", "distribution": "uniform", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 12, 12, 16]}}
?
lregularization_losses
m	variables
ntrainable_variables
o	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "leaky_re_lu_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_6", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}
?
pregularization_losses
q	variables
rtrainable_variables
s	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "up_sampling2d_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "UpSampling2D", "config": {"name": "up_sampling2d_3", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?


tkernel
ubias
vregularization_losses
w	variables
xtrainable_variables
y	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?{"name": "conv2d_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_7", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_avg", "distribution": "uniform", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 24, 24, 32]}}
?
zregularization_losses
{	variables
|trainable_variables
}	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "cropping2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Cropping2D", "config": {"name": "cropping2d_1", "trainable": true, "dtype": "float32", "cropping": {"class_name": "__tuple__", "items": [{"class_name": "__tuple__", "items": [2, 2]}, {"class_name": "__tuple__", "items": [2, 2]}]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?
~iter

beta_1
?beta_2

?decay
?learning_rate m?!m?.m?/m?<m?=m?Jm?Km?Xm?Ym?fm?gm?tm?um? v?!v?.v?/v?<v?=v?Jv?Kv?Xv?Yv?fv?gv?tv?uv?"
	optimizer
 "
trackable_list_wrapper
?
 0
!1
.2
/3
<4
=5
J6
K7
X8
Y9
f10
g11
t12
u13"
trackable_list_wrapper
?
 0
!1
.2
/3
<4
=5
J6
K7
X8
Y9
f10
g11
t12
u13"
trackable_list_wrapper
?
?layers
?non_trainable_variables
regularization_losses
	variables
 ?layer_regularization_losses
?metrics
?layer_metrics
trainable_variables
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
?non_trainable_variables
regularization_losses
	variables
 ?layer_regularization_losses
?metrics
?layer_metrics
trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
):' 2conv2d_1/kernel
: 2conv2d_1/bias
 "
trackable_list_wrapper
.
 0
!1"
trackable_list_wrapper
.
 0
!1"
trackable_list_wrapper
?
?layers
?non_trainable_variables
"regularization_losses
#	variables
 ?layer_regularization_losses
?metrics
?layer_metrics
$trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
?non_trainable_variables
&regularization_losses
'	variables
 ?layer_regularization_losses
?metrics
?layer_metrics
(trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
?non_trainable_variables
*regularization_losses
+	variables
 ?layer_regularization_losses
?metrics
?layer_metrics
,trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
):' 2conv2d_2/kernel
:2conv2d_2/bias
 "
trackable_list_wrapper
.
.0
/1"
trackable_list_wrapper
.
.0
/1"
trackable_list_wrapper
?
?layers
?non_trainable_variables
0regularization_losses
1	variables
 ?layer_regularization_losses
?metrics
?layer_metrics
2trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
?non_trainable_variables
4regularization_losses
5	variables
 ?layer_regularization_losses
?metrics
?layer_metrics
6trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
?non_trainable_variables
8regularization_losses
9	variables
 ?layer_regularization_losses
?metrics
?layer_metrics
:trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
):'2conv2d_3/kernel
:2conv2d_3/bias
 "
trackable_list_wrapper
.
<0
=1"
trackable_list_wrapper
.
<0
=1"
trackable_list_wrapper
?
?layers
?non_trainable_variables
>regularization_losses
?	variables
 ?layer_regularization_losses
?metrics
?layer_metrics
@trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
?non_trainable_variables
Bregularization_losses
C	variables
 ?layer_regularization_losses
?metrics
?layer_metrics
Dtrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
?non_trainable_variables
Fregularization_losses
G	variables
 ?layer_regularization_losses
?metrics
?layer_metrics
Htrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
):'2conv2d_4/kernel
:2conv2d_4/bias
 "
trackable_list_wrapper
.
J0
K1"
trackable_list_wrapper
.
J0
K1"
trackable_list_wrapper
?
?layers
?non_trainable_variables
Lregularization_losses
M	variables
 ?layer_regularization_losses
?metrics
?layer_metrics
Ntrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
?non_trainable_variables
Pregularization_losses
Q	variables
 ?layer_regularization_losses
?metrics
?layer_metrics
Rtrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
?non_trainable_variables
Tregularization_losses
U	variables
 ?layer_regularization_losses
?metrics
?layer_metrics
Vtrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
):'2conv2d_5/kernel
:2conv2d_5/bias
 "
trackable_list_wrapper
.
X0
Y1"
trackable_list_wrapper
.
X0
Y1"
trackable_list_wrapper
?
?layers
?non_trainable_variables
Zregularization_losses
[	variables
 ?layer_regularization_losses
?metrics
?layer_metrics
\trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
?non_trainable_variables
^regularization_losses
_	variables
 ?layer_regularization_losses
?metrics
?layer_metrics
`trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
?non_trainable_variables
bregularization_losses
c	variables
 ?layer_regularization_losses
?metrics
?layer_metrics
dtrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
):' 2conv2d_6/kernel
: 2conv2d_6/bias
 "
trackable_list_wrapper
.
f0
g1"
trackable_list_wrapper
.
f0
g1"
trackable_list_wrapper
?
?layers
?non_trainable_variables
hregularization_losses
i	variables
 ?layer_regularization_losses
?metrics
?layer_metrics
jtrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
?non_trainable_variables
lregularization_losses
m	variables
 ?layer_regularization_losses
?metrics
?layer_metrics
ntrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
?non_trainable_variables
pregularization_losses
q	variables
 ?layer_regularization_losses
?metrics
?layer_metrics
rtrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
):' 2conv2d_7/kernel
:2conv2d_7/bias
 "
trackable_list_wrapper
.
t0
u1"
trackable_list_wrapper
.
t0
u1"
trackable_list_wrapper
?
?layers
?non_trainable_variables
vregularization_losses
w	variables
 ?layer_regularization_losses
?metrics
?layer_metrics
xtrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
?non_trainable_variables
zregularization_losses
{	variables
 ?layer_regularization_losses
?metrics
?layer_metrics
|trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2iter
: (2beta_1
: (2beta_2
: (2decay
: (2learning_rate
?
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?

?total

?count
?	variables
?	keras_api"?
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
?

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "binary_accuracy"}}
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
):' 2conv2d_1/kernel/m
: 2conv2d_1/bias/m
):' 2conv2d_2/kernel/m
:2conv2d_2/bias/m
):'2conv2d_3/kernel/m
:2conv2d_3/bias/m
):'2conv2d_4/kernel/m
:2conv2d_4/bias/m
):'2conv2d_5/kernel/m
:2conv2d_5/bias/m
):' 2conv2d_6/kernel/m
: 2conv2d_6/bias/m
):' 2conv2d_7/kernel/m
:2conv2d_7/bias/m
):' 2conv2d_1/kernel/v
: 2conv2d_1/bias/v
):' 2conv2d_2/kernel/v
:2conv2d_2/bias/v
):'2conv2d_3/kernel/v
:2conv2d_3/bias/v
):'2conv2d_4/kernel/v
:2conv2d_4/bias/v
):'2conv2d_5/kernel/v
:2conv2d_5/bias/v
):' 2conv2d_6/kernel/v
: 2conv2d_6/bias/v
):' 2conv2d_7/kernel/v
:2conv2d_7/bias/v
?2?
,__inference_sequential_1_layer_call_fn_26468
,__inference_sequential_1_layer_call_fn_26554?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
 __inference__wrapped_model_26039?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *=?:
8?5
zero_padding2d_1_input?????????
?2?
G__inference_sequential_1_layer_call_and_return_conditional_losses_26328
G__inference_sequential_1_layer_call_and_return_conditional_losses_26381
G__inference_sequential_1_layer_call_and_return_conditional_losses_26669
G__inference_sequential_1_layer_call_and_return_conditional_losses_26743?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
0__inference_zero_padding2d_1_layer_call_fn_26052?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
K__inference_zero_padding2d_1_layer_call_and_return_conditional_losses_26046?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_conv2d_1_layer_call_and_return_conditional_losses_26753?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
H__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_26758?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
/__inference_max_pooling2d_1_layer_call_fn_26064?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_26058?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_conv2d_2_layer_call_and_return_conditional_losses_26768?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
H__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_26773?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
/__inference_max_pooling2d_2_layer_call_fn_26076?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
J__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_26070?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_conv2d_3_layer_call_and_return_conditional_losses_26783?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
H__inference_leaky_re_lu_3_layer_call_and_return_conditional_losses_26788?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
/__inference_max_pooling2d_3_layer_call_fn_26088?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
J__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_26082?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_conv2d_4_layer_call_and_return_conditional_losses_26798?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
H__inference_leaky_re_lu_4_layer_call_and_return_conditional_losses_26803?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
/__inference_up_sampling2d_1_layer_call_fn_26107?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
J__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_26101?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_conv2d_5_layer_call_and_return_conditional_losses_26813?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
H__inference_leaky_re_lu_5_layer_call_and_return_conditional_losses_26818?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
/__inference_up_sampling2d_2_layer_call_fn_26126?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
J__inference_up_sampling2d_2_layer_call_and_return_conditional_losses_26120?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_conv2d_6_layer_call_and_return_conditional_losses_26828?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
H__inference_leaky_re_lu_6_layer_call_and_return_conditional_losses_26833?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
/__inference_up_sampling2d_3_layer_call_fn_26145?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
J__inference_up_sampling2d_3_layer_call_and_return_conditional_losses_26139?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_conv2d_7_layer_call_and_return_conditional_losses_26844?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_cropping2d_1_layer_call_fn_26160?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
G__inference_cropping2d_1_layer_call_and_return_conditional_losses_26154?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?B?
#__inference_signature_wrapper_26595zero_padding2d_1_input"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
 __inference__wrapped_model_26039? !./<=JKXYfgtuG?D
=?:
8?5
zero_padding2d_1_input?????????
? "C?@
>
cropping2d_1.?+
cropping2d_1??????????
C__inference_conv2d_1_layer_call_and_return_conditional_losses_26753l !7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0????????? 
? ?
C__inference_conv2d_2_layer_call_and_return_conditional_losses_26768l./7?4
-?*
(?%
inputs????????? 
? "-?*
#? 
0?????????
? ?
C__inference_conv2d_3_layer_call_and_return_conditional_losses_26783l<=7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????
? ?
C__inference_conv2d_4_layer_call_and_return_conditional_losses_26798lJK7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????
? ?
C__inference_conv2d_5_layer_call_and_return_conditional_losses_26813?XYI?F
??<
:?7
inputs+???????????????????????????
? "??<
5?2
0+???????????????????????????
? ?
C__inference_conv2d_6_layer_call_and_return_conditional_losses_26828?fgI?F
??<
:?7
inputs+???????????????????????????
? "??<
5?2
0+??????????????????????????? 
? ?
C__inference_conv2d_7_layer_call_and_return_conditional_losses_26844?tuI?F
??<
:?7
inputs+??????????????????????????? 
? "??<
5?2
0+???????????????????????????
? ?
G__inference_cropping2d_1_layer_call_and_return_conditional_losses_26154?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
,__inference_cropping2d_1_layer_call_fn_26160?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
H__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_26758h7?4
-?*
(?%
inputs????????? 
? "-?*
#? 
0????????? 
? ?
H__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_26773h7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????
? ?
H__inference_leaky_re_lu_3_layer_call_and_return_conditional_losses_26788h7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????
? ?
H__inference_leaky_re_lu_4_layer_call_and_return_conditional_losses_26803h7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????
? ?
H__inference_leaky_re_lu_5_layer_call_and_return_conditional_losses_26818?I?F
??<
:?7
inputs+???????????????????????????
? "??<
5?2
0+???????????????????????????
? ?
H__inference_leaky_re_lu_6_layer_call_and_return_conditional_losses_26833?I?F
??<
:?7
inputs+??????????????????????????? 
? "??<
5?2
0+??????????????????????????? 
? ?
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_26058?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
/__inference_max_pooling2d_1_layer_call_fn_26064?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
J__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_26070?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
/__inference_max_pooling2d_2_layer_call_fn_26076?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
J__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_26082?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
/__inference_max_pooling2d_3_layer_call_fn_26088?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
G__inference_sequential_1_layer_call_and_return_conditional_losses_26328? !./<=JKXYfgtuO?L
E?B
8?5
zero_padding2d_1_input?????????
p

 
? "??<
5?2
0+???????????????????????????
? ?
G__inference_sequential_1_layer_call_and_return_conditional_losses_26381? !./<=JKXYfgtuO?L
E?B
8?5
zero_padding2d_1_input?????????
p 

 
? "??<
5?2
0+???????????????????????????
? ?
G__inference_sequential_1_layer_call_and_return_conditional_losses_26669? !./<=JKXYfgtu??<
5?2
(?%
inputs?????????
p

 
? "-?*
#? 
0?????????
? ?
G__inference_sequential_1_layer_call_and_return_conditional_losses_26743? !./<=JKXYfgtu??<
5?2
(?%
inputs?????????
p 

 
? "-?*
#? 
0?????????
? ?
,__inference_sequential_1_layer_call_fn_26468? !./<=JKXYfgtuO?L
E?B
8?5
zero_padding2d_1_input?????????
p

 
? "2?/+????????????????????????????
,__inference_sequential_1_layer_call_fn_26554? !./<=JKXYfgtuO?L
E?B
8?5
zero_padding2d_1_input?????????
p 

 
? "2?/+????????????????????????????
#__inference_signature_wrapper_26595? !./<=JKXYfgtua?^
? 
W?T
R
zero_padding2d_1_input8?5
zero_padding2d_1_input?????????"C?@
>
cropping2d_1.?+
cropping2d_1??????????
J__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_26101?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
/__inference_up_sampling2d_1_layer_call_fn_26107?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
J__inference_up_sampling2d_2_layer_call_and_return_conditional_losses_26120?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
/__inference_up_sampling2d_2_layer_call_fn_26126?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
J__inference_up_sampling2d_3_layer_call_and_return_conditional_losses_26139?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
/__inference_up_sampling2d_3_layer_call_fn_26145?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
K__inference_zero_padding2d_1_layer_call_and_return_conditional_losses_26046?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
0__inference_zero_padding2d_1_layer_call_fn_26052?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84????????????????????????????????????