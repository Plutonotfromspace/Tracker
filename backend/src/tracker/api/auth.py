"""Authentication API endpoints."""

from typing import Optional
from datetime import timedelta
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from sqlmodel import Session

from ..data.database import get_session
from ..data.user_crud import authenticate_user, update_last_login
from ..data.auth import create_access_token, verify_token, ACCESS_TOKEN_EXPIRE_MINUTES

router = APIRouter(prefix="/api/auth", tags=["authentication"])

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")


# Request/Response models
class Token(BaseModel):
    access_token: str
    token_type: str
    username: str
    full_name: Optional[str] = None
    is_demo: bool = False


class UserResponse(BaseModel):
    username: str
    email: str
    full_name: Optional[str] = None
    is_active: bool
    is_superuser: bool
    is_demo: bool


@router.post("/login", response_model=Token)
def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    session: Session = Depends(get_session),
):
    """
    Authenticate user and return JWT token.
    
    Accepts username and password via OAuth2 form.
    Returns access token on success.
    """
    user = authenticate_user(session, form_data.username, form_data.password)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Update last login time
    update_last_login(session, user)
    
    # Create access token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    
    return Token(
        access_token=access_token,
        token_type="bearer",
        username=user.username,
        full_name=user.full_name,
        is_demo=user.is_demo,
    )


@router.get("/verify", response_model=UserResponse)
def verify(
    token: str = Depends(oauth2_scheme),
    session: Session = Depends(get_session),
):
    """
    Verify JWT token and return user information.
    
    Used to check if token is still valid and get current user data.
    """
    from ..data.user_crud import get_user_by_username
    
    payload = verify_token(token)
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    username: str = payload.get("sub")
    if username is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload",
        )
    
    user = get_user_by_username(session, username)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
        )
    
    return UserResponse(
        username=user.username,
        email=user.email,
        full_name=user.full_name,
        is_active=user.is_active,
        is_superuser=user.is_superuser,
        is_demo=user.is_demo,
    )


def get_current_user(
    token: str = Depends(oauth2_scheme),
    session: Session = Depends(get_session),
):
    """
    Dependency to get current authenticated user from token.
    Use this in endpoints that need to know who the user is.
    """
    from ..data.user_crud import get_user_by_username
    from ..data.user_models import User
    
    payload = verify_token(token)
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    username: str = payload.get("sub")
    if username is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload",
        )
    
    user = get_user_by_username(session, username)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
        )
    
    return user


def require_non_demo_user(current_user = Depends(get_current_user)):
    """
    Dependency to ensure user is not a demo account.
    Use this on endpoints that should be restricted for demo users (e.g., video upload).
    """
    if current_user.is_demo:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Demo accounts cannot perform this action. This feature is restricted during the demo/development phase.",
        )
    return current_user


@router.post("/logout")
def logout():
    """
    Logout endpoint (client-side token removal).
    
    In a stateless JWT system, logout is handled client-side
    by removing the token from storage.
    """
    return {"message": "Successfully logged out"}
